import threading
import time
import torch
from wunderscout import DataExporter, HeatmapGenerator


def process_video(local_path, output_path, job_id, detector):
    print(
        f"[process_video][{time.strftime('%X')}] Job {job_id} running in thread {threading.get_ident()}"
    )
    if torch.cuda.is_available():
        print("GPU name: ", torch.cuda.get_device_name(0))

        # Run detector
        result = detector.run(local_path, output_path)

        # Generate CSVs
        DataExporter.save_csvs(result, output_path.replace(".mp4", ".csv"))

        # Generate heatmaps
        heatmap_gen = HeatmapGenerator(min_samples_for_kde=10)

        # Get all player IDs and generate their heatmaps
        all_player_ids = result.get_all_player_ids()

        for player_id in all_player_ids:
            try:
                heatmap_data = heatmap_gen.generate_player_heatmap(
                    result, player_id, method="both"
                )

                # Save histogram (if available)
                if "histogram" in heatmap_data:
                    histogram_path = (
                        f"/tmp/{job_id}/heatmap/player{player_id}_histogram.json"
                    )
                    heatmap_gen.save_heatmap(heatmap_data["histogram"], histogram_path)
                else:
                    print(f"No histogram data for player {player_id}")

                # Save KDE (if available)
                if "kde" in heatmap_data:
                    kde_path = f"/tmp/{job_id}/heatmap/player{player_id}_kde.json"
                    heatmap_gen.save_heatmap(heatmap_data["kde"], kde_path)
                else:
                    print(
                        f"No KDE data for player {player_id} (insufficient samples/variation)"
                    )

            except ValueError as e:
                print(
                    f"Warning: Could not generate heatmap for player {player_id}: {e}"
                )

        return result

    else:
        print("No GPU Access")
