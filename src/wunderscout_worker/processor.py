import os
import json
import asyncio
import aio_pika
import aiofiles
import threading
from datetime import datetime, timezone
from .lib import process_context, get_redis
from .engine import process_video
from wunderscout import ScoutingPipeline


async def process_message(message, ctx, pipeline):
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

        # Yolo Detection
        try:
            await asyncio.to_thread(
                process_video, local_path, output_path, job_id, pipeline
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
            print("WORKER: Uploading results to S3 ...")
            output_key = f"results/{job_id}.mp4"
            async with aiofiles.open(output_path, "rb") as f:
                file_data = await f.read()
                await s3.put_object(Bucket=bucket_name, Key=output_key, Body=file_data)

            histogram_json_path = (
                "/app/src/wunderscout_worker/heatmap/player17_histogram.json"
            )
            histogram_key = f"results/{job_id}_histogram.json"

            kde_json_path = "/app/src/wunderscout_worker/heatmap/player17_kde.json"
            kde_key = f"results/{job_id}_kde.json"

            passnetwork_json_path = (
                "/app/src/wunderscout_worker/pass_network/pass_network.json"
            )
            passnetwork_key = f"results/{job_id}_passnetwork.json"

            async with aiofiles.open(histogram_json_path, "rb") as f:
                file_data = await f.read()
                await s3.put_object(
                    Bucket=bucket_name,
                    Key=histogram_key,
                    Body=file_data,
                )

            async with aiofiles.open(kde_json_path, "rb") as f:
                file_data = await f.read()
                await s3.put_object(
                    Bucket=bucket_name,
                    Key=kde_key,
                    Body=file_data,
                )

            async with aiofiles.open(passnetwork_json_path, "rb") as f:
                file_data = await f.read()
                await s3.put_object(
                    Bucket=bucket_name,
                    Key=passnetwork_key,
                    Body=file_data,
                )

            # Postgres Metadata
            # CREATE TABLE jobs (
            #   id UUID PRIMARY KEY,
            #   s3_key TEXT NOT NULL,
            #   status TEXT NOT NULL DEFAULT 'pending',
            #   submitted_at TIMESTAMPTZ DEFAULT now(),
            #   started_at TIMESTAMPTZ,
            #   finished_at TIMESTAMPTZ
            # );
            #
            # CREATE TABLE job_results (
            #   id SERIAL PRIMARY KEY,
            #   job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            #   s3_key TEXT NOT NULL,
            #   created_at TIMESTAMPTZ DEFAULT now()
            # );
            #
            # CREATE TABLE job_artifacts (
            #   id SERIAL PRIMARY KEY,
            #   job_id UUID NOT NULL REFERENCES jobs(id),
            #   artifact_name TEXT,
            #   artifact_type TEXT,
            #   s3_key TEXT NOT NULL,
            #   created_at TIMESTAMPTZ DEFAULT now()
            # );
            print("WORKER: Uploading metadata to postgres ...")
            async with ctx["pg"].connection() as conn:
                async with conn.transaction():
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "UPDATE jobs SET status = %s, finished_at = %s WHERE id = %s",
                            ("completed", finish_time, job_id),
                        )

                        await cur.execute(
                            "INSERT INTO job_results (job_id, s3_key) VALUES (%s, %s)",
                            (job_id, output_key),
                        )

                        await cur.execute(
                            "INSERT INTO job_artifacts (job_id, artifact_name, artifact_type, s3_key) VALUES (%s, %s, %s, %s)",
                            (job_id, "histogram", "json", histogram_key),
                        )

                        await cur.execute(
                            "INSERT INTO job_artifacts (job_id, artifact_name, artifact_type, s3_key) VALUES (%s, %s, %s, %s)",
                            (job_id, "kde", "json", kde_key),
                        )

                        await cur.execute(
                            "INSERT INTO job_artifacts (job_id, artifact_name, artifact_type, s3_key) VALUES (%s, %s, %s, %s)",
                            (job_id, "passnetwork", "json", passnetwork_key),
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
        pipeline = ScoutingPipeline(player_weights="", field_weights="")
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
                        await process_message(message, ctx, pipeline)
                    except Exception as e:
                        print(f"Worker {process_id}: Error processing message: {e}")
                        import traceback

                        traceback.print_exc()

                await queue.consume(on_message)
                await asyncio.Future()

        except Exception as e:
            raise e
