import os
from dotenv import load_dotenv
import multiprocessing
import asyncio
from .processor import wrap_context

load_dotenv()


async def async_worker(process_id):
    """Main async function for each worker process"""
    await wrap_context(process_id)


def start_process(process_id):
    """Entry point for each worker process"""
    print(f"Process {process_id} starting with PID: {os.getpid()}")
    try:
        asyncio.run(async_worker(process_id))
    except KeyboardInterrupt:
        print(f"Process {process_id}: Interrupted.")
    except Exception as e:
        print(f"Process {process_id}: Fatal error: {e}")
        raise e


def main():
    NUM_WORKER_PROCESSES = 4
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        print("Spawn start method already set.")
        pass

    print(f"Main process (PID {os.getpid()}): Starting with 'spawn' method.")

    processes = []

    # Start the processes
    print(f"Main process: Starting {NUM_WORKER_PROCESSES} processes ...")
    for i in range(NUM_WORKER_PROCESSES):
        p = multiprocessing.Process(target=start_process, args=(i,))
        processes.append(p)
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Main process: KeyboardInterrupt (ctrl + c). Shutting down ...)")
        for p in processes:
            if p.is_alive():
                print(f"Main process: Process {p.pid} terminating ...")
                p.terminate()
                p.join(timeout=2)
                if p.is_alive():
                    print(
                        f"Main process: Process {p.pid} did not terminate gracefully."
                    )
    except Exception as e:
        print(f"Main process: Unexpected error: {e}.")
    finally:
        print("Main process: All child processes have been managed.")
        print("Main process: Exiting ...")


if __name__ == "__main__":
    print("Inside the main guard.")
    main()
