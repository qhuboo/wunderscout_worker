import modal

app = modal.app("wunderscout-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("torch", "wunderscout")
)

volume = modal.Volume.from_name("model-weights", create_if_missing=True)


@app.function(
    image="A10G",
    timeout=1800,
    volumes={"/app/data/models": volume},
    secrets=[modal.Secret.from_name("aws-credentials")],
)
def run_detection():
    """
    Downloads video, runs detection, generates artifacts,
    uploads everything to S3, and returns metadata.
    """
