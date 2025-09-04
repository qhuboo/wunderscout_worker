import os
import json
import pika
import boto3
from dotenv import load_dotenv

from .yolo import process_video

load_dotenv()

# Create a single S3 client
s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

class WunderScoutWorker:
    def __init__(self):
        print("Initializing WunderScout Worker...")

        # RabbitMQ Connection
        self.rabbitmq_url = os.getenv("RABBITMQ_URL")
        print(f"WORKER: RabbitMQ URL: {self.rabbitmq_url}")


    def start_consuming(self):
        print("WORKER: Connecting to RabbitMQ...")
        try:
            # Connect to RabbitMQ
            connection = pika.BlockingConnection(pika.ConnectionParameters(host="rabbitmq"))
            channel = connection.channel()

            queue_name = "test_queue"
            channel.queue_declare(queue=queue_name, durable=True)


            channel.basic_consume(queue=queue_name, on_message_callback=self.process_message)

            print(f"WORKER: Waiting for messages on '{queue_name}'...")
            
            channel.start_consuming()

        except KeyboardInterrupt:
            print("WORKER: Stopping worker...")
            channel.stop_consuming()
            connection.close()

        except Exception as e:
            print(f"WORKER: Connection error: {e}")

    def process_message(self, ch, method, properties, body):
        try:
            message = json.loads(body.decode('utf-8'))
            print(f"WORKER: Received: {message}")
            
            job_id = message.get('jobId')
            key = message.get('key')

            local_path = os.path.join("/tmp", key) 
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            s3.download_file(os.getenv("S3_BUCKET"), key, local_path)
            
            print(f"WORKER: Processing job {job_id} with key {key}")

            process_video(job_id, 3)

            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            print(f"WORKER: Error: {e}")