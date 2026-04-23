import os
import time
import json
import asyncio
import aio_pika
import threading
from datetime import datetime, timezone
from .lib import process_context, get_redis
import modal


async def process_message(message, ctx):
    async with message.process():
        body_text = message.body.decode()
        body = json.loads(body_text)
        job_id = body["jobId"]
        key = body["key"]

        if job_id is None or key is None:
            raise Exception("Job id and key are required.")

        start_time = datetime.now(timezone.utc)

        async with get_redis(ctx) as redis:
            message = {
                "jobId": str(job_id),
                "status": "started",
                "timestamp": start_time.isoformat(),
            }
            await redis.publish("job_updates", json.dumps(message))

        async with ctx["pg"].connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO jobs (id, s3_key, started_at) VALUES (%s, %s, %s)",
                    (job_id, key, start_time),
                )

        print(
            f"WORKER[process_message][{start_time}] Started Job: {job_id} PID={os.getpid()} thread id={threading.get_ident()}, name={threading.current_thread().name}"
        )

        ################################ Modal Function #################################################
        #                                                                                                       #
        #                                                                                                       #

        async with get_redis(ctx) as redis:
            message = {
                "jobId": job_id,
                "status": "sent to modal",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await redis.publish("job_updates", json.dumps(message))

        LoadModels = modal.Cls.from_name(
            "wunderscout-inference",
            "LoadModels",
        )

        print("Starting modal function.")
        result = await LoadModels().run_detection.remote.aio(job_id)

        print("Modal result: ", result.keys())

        async with get_redis(ctx) as redis:
            message = {
                "jobId": job_id,
                "status": "modal function done",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await redis.publish("job_updates", json.dumps(message))

        #                                                                                                       #
        #                                                                                                       #
        ##########################################################################################################

        finish_time = datetime.now(timezone.utc)

        print("WORKER[process_message]: Uploading data to postgres ...")

        print("Job Id: ", result["job_id"])
        print("Game Id: ", result["game_id"])

        print(type(result["annotated_video"]))
        print(str(result["annotated_video"]))

        async with ctx["pg"].connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    # CREATE TABLE jobs (
                    #   id UUID PRIMARY KEY,
                    #   s3_key TEXT NOT NULL,
                    #   status TEXT NOT NULL DEFAULT 'pending',
                    #   submitted_at TIMESTAMPTZ DEFAULT now(),
                    #   started_at TIMESTAMPTZ,
                    #   finished_at TIMESTAMPTZ
                    # );
                    await cur.execute(
                        "UPDATE jobs SET status = %s, finished_at = %s WHERE id = %s",
                        ("completed", finish_time, job_id),
                    )

                    # CREATE TABLE annotated_videos (
                    #   id SERIAL PRIMARY KEY,
                    #   job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    #   s3_key TEXT NOT NULL,
                    #   created_at TIMESTAMPTZ DEFAULT now()
                    # );
                    await cur.execute(
                        "INSERT INTO annotated_videos (job_id, s3_key) VALUES (%s, %s)",
                        (job_id, str(result["annotated_video"])),
                    )

                    # CREATE TABLE games (
                    #   id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    #   job_id UUID NOT NULL REFERENCES jobs(id),
                    #   created_at TIMESTAMPTZ DEFAULT now()
                    # );

                    # Insert game and get game_id
                    game_id = result["game_id"]
                    await cur.execute(
                        "INSERT INTO games (id, job_id) VALUES (%s, %s)",
                        (game_id, job_id),
                    )

                    # CREATE TABLE teams (
                    #   id SERIAL PRIMARY KEY,
                    #   game_id UUID NOT NULL REFERENCES games(id) ON DELETE CASCADE,
                    #   detector_team_id INTEGER NOT NULL,
                    #   name TEXT NOT NULL,
                    #   csv_s3_key TEXT NOT NULL,
                    #
                    #   UNIQUE(game_id, detector_team_id)
                    # );

                    # Insert teams
                    team_id_map = {}  # Maps detector_team_id -> db team_id
                    for detector_team_id in result["teams"].keys():
                        print(
                            f"Team keys for {detector_team_id}: ",
                            result["teams"][detector_team_id].keys(),
                        )
                        await cur.execute(
                            "INSERT INTO teams (game_id, detector_team_id, name) VALUES (%s, %s, %s) RETURNING id",
                            (
                                game_id,
                                detector_team_id,
                                f"Team {int(detector_team_id) + 1}",
                            ),
                        )

                        team_id = (await cur.fetchone())[0]
                        team_id_map[detector_team_id] = team_id

                        # CREATE TABLE team_analytics (
                        #   id SERIAL PRIMARY KEY,
                        #   team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
                        #   artifact_type TEXT NOT NULL,
                        #   s3_key TEXT NOT NULL,
                        #   created_at TIMESTAMPTZ DEFAULT now()
                        # );
                        for artifact in result["teams"][detector_team_id].keys():
                            await cur.execute(
                                "INSERT INTO team_analytics (team_id, artifact_type, s3_key) VALUES (%s, %s, %s)",
                                (
                                    team_id,
                                    artifact,
                                    str(result["teams"][detector_team_id][artifact]),
                                ),
                            )

                    print("Team id map: ", team_id_map)

                    # CREATE TABLE players (
                    #   id SERIAL PRIMARY KEY,
                    #   game_id UUID NOT NULL REFERENCES games(id) ON DELETE CASCADE,
                    #   tracker_id INTEGER NOT NULL,
                    #   team_id INTEGER NOT NULL REFERENCES teams(id),
                    #   name TEXT,
                    #
                    #   UNIQUE(game_id, tracker_id)
                    # );
                    #
                    # CREATE TABLE player_analytics (
                    #   id SERIAL PRIMARY KEY,
                    #   player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
                    #   artifact_type TEXT NOT NULL,
                    #   s3_key TEXT NOT NULL,
                    #   created_at TIMESTAMPTZ DEFAULT now()
                    # );

                    # Insert all players and analytics
                    for tracker_id in result["players"].keys():
                        print("player keys: ", result["players"][tracker_id].keys())
                        print(
                            "player team id: ", result["players"][tracker_id]["team_id"]
                        )
                        await cur.execute(
                            "INSERT INTO players (game_id, tracker_id, team_id, name) VALUES (%s, %s, %s, %s) RETURNING id",
                            (
                                game_id,
                                tracker_id,
                                team_id_map[
                                    str(result["players"][tracker_id]["team_id"])
                                ],
                                f"Player {tracker_id}",
                            ),
                        )
                        player_id = (await cur.fetchone())[0]
                        if result["players"][tracker_id].get("kde") is not None:
                            print("KDE: ", result["players"][tracker_id].get("kde"))
                            await cur.execute(
                                "INSERT INTO player_analytics (player_id, artifact_type, s3_key) VALUES (%s, %s, %s)",
                                (
                                    player_id,
                                    "kde",
                                    str(result["players"][tracker_id]["kde"]),
                                ),
                            )
                        if result["players"][tracker_id].get("histogram") is not None:
                            print(
                                "Histogram: ",
                                result["players"][tracker_id].get("histogram"),
                            )
                            await cur.execute(
                                "INSERT INTO player_analytics (player_id, artifact_type, s3_key) VALUES (%s, %s, %s)",
                                (
                                    player_id,
                                    "histogram",
                                    str(result["players"][tracker_id]["histogram"]),
                                ),
                            )

        async with get_redis(ctx) as redis:
            message = {
                "jobId": job_id,
                "status": "completed",
                "gameId": str(game_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await redis.publish("job_updates", json.dumps(message))

        return {"job_id": job_id}


async def wrap_context(process_id):
    print(f"WORKER[process: {process_id}][wrap_context]")
    async with process_context() as ctx:
        print(f"WORKER[process: {process_id}][wrap_context][process_context]")
        rabbitmq_url = os.getenv("RABBITMQ_URL")
        if rabbitmq_url is None:
            raise RuntimeError(
                f"WORKER[process: {process_id}][wrap_context][process_context][RuntimeError]: RABBITMQ_URL env variable is required."
            )
        try:
            # RabbitMQ
            connection = await aio_pika.connect_robust(rabbitmq_url, heartbeat=3600)
            async with connection:
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=2)
                queue = await channel.declare_queue("worker_queue", durable=True)

                async def on_message(message):
                    print(
                        f"WORKER[process: {process_id}][wrap_context][process_context][on_message]"
                    )
                    try:
                        await process_message(message, ctx)
                    except Exception as e:
                        print(
                            f"WORKER[process: {process_id}][on_message][Exception]: Error processing message: {e}"
                        )
                        import traceback

                        traceback.print_exc()

                await queue.consume(on_message)
                await asyncio.Future()

        except Exception as e:
            raise e
