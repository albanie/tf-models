import argparse
from pathlib import Path
from google.cloud import storage
from os.path import join as pjoin


def fetch_ckpts_with_prefix(bucket_name, dest_dir, prefix, key_path):
    """Fetch checkpoints from Google Cloud Storage.
    """
    storage_client = storage.Client.from_service_account_json(key_path)
    bucket = storage_client.get_bucket(bucket_name)

    # use list to ensure reusable generator
    blobs = list(bucket.list_blobs(prefix=prefix, delimiter=None))

    blob_names = [Path(blob.name).name for blob in blobs]
    assert len(blob_names) == 3, "expected to find 3 files per checkpoint"
    for blob, blob_name in zip(blobs, blob_names):
        dest_path = Path(dest_dir) / blob_name
        if not dest_path.exists():
            print("downloading {} to {}".format(blob_name, dest_path))
            blob.download_to_filename(str(dest_path))
        else:
            print("found existing {} at {}, skipping".format(blob_name, dest_path))


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default="ckpts")
parser.add_argument("--bucket", type=str, default="imagenet-dev-eu")
parser.add_argument("--api_key", type=str, default=Path.home() / "keys/bucket-key.json")
args = parser.parse_args()

ckpt_list = [
    ("exp-v2/talbot/prof-n1-50-float32-talbot", "model.ckpt-112590"),
]


for folder, fname in ckpt_list:
    # training uses the convention that the parent directory is descriptive of the model
    model_desc = folder.split("/")[-1]
    # use os.path.join rather than Pathlib to avoid mangling bucket name
    src_prefix = pjoin(folder, fname)

    dest_dir = Path(args.ckpt_dir) / model_desc
    if not Path(dest_dir).exists():
        dest_dir.mkdir(exist_ok=True, parents=True)
    fetch_ckpts_with_prefix(
        bucket_name=args.bucket,
        dest_dir=dest_dir,
        prefix=src_prefix,
        key_path=str(args.api_key),
    )
