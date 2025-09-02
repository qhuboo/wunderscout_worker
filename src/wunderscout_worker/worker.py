import os
import json
import pika
import boto3
from dotenv import load_dotenv

load_dotenv()

class WunderScoutWorker:
    def __init__(self):
        print("Initializing WunderScout Worker...")
        print("=== Environment Variables ===")
        print(f"RABBITMQ_URL: {os.getenv('RABBITMQ_URL')}")
        print(f"AWS_ACCESS_KEY_ID: {os.getenv('AWS_ACCESS_KEY_ID')}")
        print(f"AWS_SECRET_ACCESS_KEY: {os.getenv('AWS_SECRET_ACCESS_KEY')}")
        print(f"AWS_REGION: {os.getenv('AWS_REGION')}")
        print("=============================")

    def start_consuming(self):
        print("Connecting to RabbitMQ...")