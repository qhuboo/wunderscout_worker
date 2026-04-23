from collections import defaultdict
import os
import subprocess
import modal

ENV = os.environ.get("ENV", "development")


def get_wunderscout_dep():
    if ENV == "production":
        print("Installing v0.2.2")
        return "wunderscout==0.2.2"

    # For when it runs in the container
    repo_path = os.path.expanduser(
        "~/Documents/dev/github_repos/wunderscout_workspace/wunderscout"
    )
    if not os.path.exists(repo_path):
        return "wunderscout"

    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True,
        cwd=os.path.expanduser(
            "~/Documents/dev/github_repos/wunderscout_workspace/wunderscout"
        ),
    )
    branch = result.stdout.strip()
    print(
        f"Installing: wunderscout @ git+https://github.com/qhuboo/wunderscout.git@{branch}"
    )
    return f"wunderscout @ git+https://github.com/qhuboo/wunderscout.git@{branch}"


app = modal.App("wunderscout-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(get_wunderscout_dep(), force_build=True)
)

volume = modal.Volume.from_name("wunderscout")

s3_mount = modal.CloudBucketMount(
    bucket_name=os.environ.get("S3_BUCKET", ""),
    secret=modal.Secret.from_name("aws-secret"),
    read_only=False,
)


@app.cls(
    image=image,
    gpu="A10",
    timeout=1800,
    volumes={
        "/s3-bucket": s3_mount,
        "/wunderscout": volume,
    },
)
class LoadModels:
    @modal.enter()
    def load_models(self):
        import wunderscout

        self.models = wunderscout.Models(
            player_weights="/wunderscout/player.pt",
            field_weights="/wunderscout/field.pt",
            siglip_path="/wunderscout/siglip_vision",
        )

    @modal.method()
    def run_detection(self, job_id: str) -> dict:
        import shutil
        import uuid
        from pathlib import Path
        import logging
        from typing import Any
        import wunderscout

        wunderscout.set_stream_logger(level=logging.INFO)

        output_dir = Path(f"/tmp/{job_id}/")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download s3 bucket video
        # creating s3 path
        s3_video_key = f"jobs/{job_id}/input/{job_id}.mp4"
        s3_video_path = Path(f"/s3-bucket/{s3_video_key}")

        # Check to see if the video exists in s3
        if not s3_video_path.exists():
            return {"job_id": job_id, "error": "not found"}

        # Video will be downloaded into /tmp/{job_id}/input/
        video_dir = output_dir / "input"
        video_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(s3_video_path, video_dir)

        # Detection
        # Annotated video will be saved at /tmp/{job_id}/output/
        annotated_video_dir = output_dir / "output"
        annotated_video_dir.mkdir(parents=True, exist_ok=True)

        # Return object
        game_id = str(uuid.uuid4())
        job: dict[str, Any] = {"job_id": job_id, "game_id": game_id}

        detector = wunderscout.Detector(self.models)
        try:
            frames = detector.run(video_dir / f"{job_id}.mp4", annotated_video_dir)
            # Creating the results path in s3 for annotated video
            s3_results_path = Path(f"/s3-bucket/jobs/{job_id}/output/")
            s3_results_path.mkdir(parents=True, exist_ok=True)

            # Uploading annotated video
            annotated_video_path = Path(annotated_video_dir / f"{job_id}_annotated.mp4")
            if annotated_video_path.is_file():
                shutil.copy(annotated_video_path, s3_results_path)

            job["annotated_video"] = (
                Path(str(s3_results_path).replace("/s3-bucket", ""))
                / f"{job_id}_annotated.mp4"
            )

            # Generating team heatmaps
            paths = []
            heatmaps_dir = output_dir / "heatmaps"
            generator = wunderscout.HeatmapGenerator()
            team_ids = frames.get_all_team_ids()
            for team_id in team_ids:
                heatmap = generator.team(frames, team_id)
                result = heatmap.save(heatmaps_dir)
                paths.extend(result.successful_paths)

            player_ids = frames.get_all_player_ids()
            for player_id in player_ids:
                heatmap = generator.player(frames, player_id)
                result = heatmap.save(heatmaps_dir)
                paths.extend(result.successful_paths)

            # Uploading heatmaps
            s3_keys = []
            teams = defaultdict(dict)
            players = defaultdict(dict)
            s3_heatmaps_dir = Path(
                f"/s3-bucket/jobs/{job_id}/games/{game_id}/heatmaps/"
            )
            s3_heatmaps_dir.mkdir(parents=True, exist_ok=True)
            for file in list(heatmaps_dir.iterdir()):
                if file in paths:
                    filename = file.name
                    parts = Path(filename).stem.split("_")

                    prefix = parts[0]
                    identifier = parts[1]
                    heatmap_type = parts[2]

                    s3_key = s3_heatmaps_dir / filename
                    shutil.copy(file, s3_heatmaps_dir)
                    clean_s3_key = str(s3_key).replace("/s3-bucket/", "")
                    s3_keys.append(clean_s3_key)

                    if prefix == "team":
                        teams[identifier][heatmap_type] = clean_s3_key
                    if prefix == "player":
                        players[identifier][heatmap_type] = clean_s3_key
                        players[identifier]["team_id"] = frames.get_team_for_player(
                            int(identifier)
                        )

            # Generating CSVs
            csv_dir = output_dir / "csvs"
            csv_dir.mkdir(parents=True, exist_ok=True)
            result = frames.save_csvs(csv_dir)
            s3_csv_dir = Path(f"/s3-bucket/jobs/{job_id}/games/{game_id}/csvs/")
            s3_csv_dir.mkdir(parents=True, exist_ok=True)
            for file in result.successful_paths:
                filename = file.name
                parts = Path(filename).stem.split("_")
                team_id = parts[1]
                shutil.copy(file, s3_csv_dir)

                teams[team_id]["csv"] = str(s3_csv_dir / filename).replace(
                    "/s3-bucket/", ""
                )

            job["teams"] = teams
            job["players"] = players

            return job
        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"Error": e}


@app.local_entrypoint()
def main():
    models = LoadModels()
    models.run_detection.remote("17e9d1f0-6963-4ec3-8239-5947ae41577e")
