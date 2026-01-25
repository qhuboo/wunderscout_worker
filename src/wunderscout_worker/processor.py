import os
import json
import asyncio
import aio_pika
import aiofiles
import threading
from pathlib import Path
from datetime import datetime, timezone
from .lib import process_context, get_redis
from .engine import process_video
from wunderscout import Detector


async def process_message(message, ctx, detector):
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
                "jobId": job_id,
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
            f"WORKER:[process_message][{start_time}] Started Job: {job_id} PID={os.getpid()} thread id={threading.get_ident()}, name={threading.current_thread().name}"
        )

        # Download S3
        bucket_name = os.getenv("S3_BUCKET")

        if bucket_name is None:
            raise Exception("Bucket name is required.")

        local_path = f"/tmp/{job_id}.mp4"
        output_path = f"/tmp/{job_id}_processed.mp4"

        s3 = ctx["s3"]
        s3_obj = await s3.get_object(Bucket=bucket_name, Key=key)
        async with aiofiles.open(local_path, "wb") as f:
            stream = s3_obj["Body"]
            while file_data := await stream.read(1024 * 1024):
                await f.write(file_data)

        async with get_redis(ctx) as redis:
            message = {
                "jobId": job_id,
                "status": "detection",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await redis.publish("job_updates", json.dumps(message))

        # Detection
        try:
            result = await asyncio.to_thread(
                process_video, local_path, output_path, job_id, detector
            )

            finish_time = datetime.now(timezone.utc)
            print(
                f"WORKER:[process_message][{finish_time}] Finished Job: {job_id} PID={os.getpid()}"
            )

            async with get_redis(ctx) as redis:
                message = {
                    "jobId": job_id,
                    "status": "uploading results",
                    "timestamp": finish_time.isoformat(),
                }
                await redis.publish("job_updates", json.dumps(message))

            # S3 upload
            print("WORKER[process_message]: Uploading results to S3 ...")
            output_key = f"results/{job_id}_processed.mp4"
            async with aiofiles.open(output_path, "rb") as f:
                file_data = await f.read()
                await s3.put_object(Bucket=bucket_name, Key=output_key, Body=file_data)

            # Upload all data files
            heatmap_dir = Path(f"/tmp/{job_id}/heatmap")

            if heatmap_dir.exists():
                print(
                    "WORKER[process_message][uploading artifacts]: heatmap_dir exists"
                )
                heatmap_files = list(heatmap_dir.glob("*.json"))
                print(
                    f"WORKER[process_message]: uploading {len(heatmap_files)} heatmap files."
                )

                for heatmap_file in heatmap_files:
                    s3_key = f"results/{job_id}/heatmap/{heatmap_file.name}"

                    async with aiofiles.open(heatmap_file, "rb") as f:
                        file_data = await f.read()
                        await s3.put_object(
                            Bucket=bucket_name,
                            Key=s3_key,
                            Body=file_data,
                        )
            else:
                print(
                    "WORKER[process_message][uploading artifacts]: heatmap_dir did not exist"
                )

            # CREATE TABLE jobs (
            #   id UUID PRIMARY KEY,
            #   s3_key TEXT NOT NULL,
            #   status TEXT NOT NULL DEFAULT 'pending',
            #   submitted_at TIMESTAMPTZ DEFAULT now(),
            #   started_at TIMESTAMPTZ,
            #   finished_at TIMESTAMPTZ
            # );
            #
            # CREATE TABLE annotated_videos (
            #   id SERIAL PRIMARY KEY,
            #   job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            #   s3_key TEXT NOT NULL,
            #   created_at TIMESTAMPTZ DEFAULT now()
            # );
            #
            # CREATE TABLE games (
            #   id UUID PRIMARY KEY,
            #   job_id UUID NOT NULL REFERENCES jobs(id),
            #   created_at TIMESTAMPTZ DEFAULT now()
            # );
            #
            # CREATE TABLE teams (
            #   id SERIAL PRIMARY KEY,
            #   game_id UUID NOT NULL REFERENCES games(id) ON DELETE CASCADE,
            #   detector_team_id INTEGER NOT NULL,
            #   team_name TEXT NOT NULL,
            #
            #   UNIQUE(game_id, detector_team_id)
            # );
            #
            # CREATE TABLE players (
            #   id SERIAL PRIMARY KEY,
            #   game_id UUID NOT NULL REFERENCES games(id) ON DELETE CASCADE,
            #   tracker_id INTEGER NOT NULL,
            #   team_id INTEGER NOT NULL REFERENCES teams(id),
            #   display_name TEXT,
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

            print("WORKER[process_message]: Uploading data to postgres ...")
            async with ctx["pg"].connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "UPDATE jobs SET status = %s, finished_at = %s WHERE id = %s",
                            ("completed", finish_time, job_id),
                        )

                        await cur.execute(
                            "INSERT INTO annotated_videos (job_id, s3_key) VALUES (%s, %s)",
                            (job_id, output_key),
                        )

                        # Insert game and get game_id
                        await cur.execute(
                            "INSERT INTO games (job_id) VALUES (%s)",
                            (job_id),
                        )
                        game_id = (await cur.fetchone())[0]

                        # Insert teams and get db team ids
                        team_ids = {}  # Maps detector_team_id -> db team id
                        for detector_team_id in [0, 1]:
                            await cur.execute(
                                "INSERT INTO teams (game_id, detector_team_id, team_name) VALUES (%s, %s, %s) RETURNING id",
                                (
                                    game_id,
                                    detector_team_id,
                                    f"Team {detector_team_id + 1}",
                                ),
                            )
                            team_ids[detector_team_id] = (await cur.fetchone())[0]

                        # Insert player analytics
                        for (
                            tracker_id,
                            detector_team_id,
                        ) in result.team_assignments.items():
                            db_team_id = team_ids[detector_team_id]
                            await cur.execute(
                                "INSERT INTO players (game_id, tracker_id, team_id) VALUES (%s, %s, %s) RETURNING id",
                                (game_id, tracker_id, db_team_id),
                            )
                            db_player_id = (await cur.fetchone())[0]

                            if heatmap_dir.exists():
                                for heatmap_file in heatmap_dir.glob("*.json"):
                                    # Parse filename get type
                                    # Format: player{id}_{histogram | kde}.json
                                    filename = heatmap_file.stem
                                    parts = filename.split("_")

                                    if len(parts) == 2:
                                        artifact_type = parts[1]  # "histogram" or "kde"
                                        s3_key = f"results/{job_id}/heatmaps/{heatmap_file.name}"

                                        await cur.execute(
                                            "INSERT INTO player_analytics (player_id, artifact_type, s3_key) VALUES (%s, %s, %s)",
                                            (
                                                db_player_id,
                                                artifact_type,
                                                s3_key,
                                            ),
                                        )

            async with get_redis(ctx) as redis:
                message = {
                    "jobId": job_id,
                    "status": "completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await redis.publish("job_updates", json.dumps(message))
        finally:
            for path in [local_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)


async def wrap_context(process_id):
    async with process_context() as ctx:
        detector = Detector(
            player_weights="/app/data/models/player_detection.pt",
            field_weights="/app/data/models/field_keypoint_detection.pt",
        )
        rabbitmq_url = os.getenv("RABBITMQ_URL")
        if rabbitmq_url is None:
            raise RuntimeError("RABBITMQ_URL env variable is required.")
        try:
            # RabbitMQ
            connection = await aio_pika.connect_robust(rabbitmq_url, heartbeat=3600)
            async with connection:
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=2)
                queue = await channel.declare_queue("worker_queue", durable=True)

                async def on_message(message):
                    try:
                        await process_message(message, ctx, detector)
                    except Exception as e:
                        print(f"Worker {process_id}: Error processing message: {e}")
                        import traceback

                        traceback.print_exc()

                await queue.consume(on_message)
                await asyncio.Future()

        except Exception as e:
            raise e
