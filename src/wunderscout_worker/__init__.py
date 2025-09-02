from .worker import WunderScoutWorker
	
def main():
	worker = WunderScoutWorker()
	worker.start_consuming()
	
__all__ = ["WunderScoutWorker", "main"]