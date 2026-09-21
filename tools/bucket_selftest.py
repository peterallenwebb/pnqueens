#!/usr/bin/env python3
"""Checks the records bucket and the signing key before a run: the bucket's
settings (versioning on, public access prevented), a signed PUT of a small
object, its read-back through the API, its listing, and that overwriting
it with different contents of the same length is seen by the sync.

    bucket_selftest.py [--bucket B] [--key FILE]
"""
import argparse, os, sys, tempfile, time
import gcs

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--bucket", default="pnqueens-compute-records")
ap.add_argument("--key", default=os.path.join(HERE, "..", "..", "nqueens-signer.key"))
a = ap.parse_args()

key = gcs.load_key(a.key)
print(f"key: {key['client_email']}")
token = gcs.access_token(key)
settings = gcs.bucket_settings(token, a.bucket)
versioning = settings.get("versioning", {}).get("enabled", False)
pap = settings.get("iamConfiguration", {}).get("publicAccessPrevention")
uniform = settings.get("iamConfiguration", {}).get("uniformBucketLevelAccess", {}).get("enabled", False)
print(f"bucket gs://{a.bucket}: location {settings.get('location')}, versioning {'on' if versioning else 'OFF'}, "
      f"public access prevention {pap}, uniform access {'on' if uniform else 'OFF'}")
problems = []
if not versioning:
    problems.append("object versioning is off (an overwritten record could not be recovered)")
if pap != "enforced":
    problems.append("public access prevention is not enforced")

obj = f"selftest/{int(time.time())}.txt"
body = f"pnqueens bucket self-test {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(body)
    path = f.name
ok, msg = gcs.curl_put(gcs.sign_url(key, a.bucket, obj, expires=600), path)
print(f"signed PUT of {obj}: {'ok' if ok else 'FAILED ' + msg}")
if not ok:
    os.unlink(path)
    problems.append("the signed upload failed")
else:
    back = gcs.api_get(token, f"{gcs.STORAGE}/storage/v1/b/{a.bucket}/o/{obj.replace('/', '%2F')}?alt=media").decode()
    print(f"read-back: {'ok' if back == body else 'MISMATCH'}")
    if back != body:
        problems.append("the object read back differs")
    listed = {n: md5 for n, _, md5 in gcs.list_objects(token, a.bucket, "selftest/")}
    print(f"listing: {len(listed)} self-test objects{', includes it' if obj in listed else ', DOES NOT include it'}")
    if obj not in listed:
        problems.append("the listing does not show the uploaded object")
    else:
        # The sync's test: a local copy, then the object overwritten with
        # different contents of the same length, must be fetched again.
        local = os.path.join(tempfile.mkdtemp(), "obj")
        gcs.download(token, a.bucket, obj, local, listed[obj])
        same_after = gcs.needs_fetch(local, listed[obj])
        body2 = body[:-2] + "X\n"
        with open(path, "w") as f:
            f.write(body2)
        ok2, msg2 = gcs.curl_put(gcs.sign_url(key, a.bucket, obj, expires=600), path)
        os.unlink(path)
        md5_2 = {n: md5 for n, _, md5 in gcs.list_objects(token, a.bucket, "selftest/")}.get(obj)
        changed = ok2 and gcs.needs_fetch(local, md5_2)
        print(f"same-length overwrite seen by the sync: {'ok' if not same_after and changed else 'FAILED'}")
        if same_after or not changed:
            problems.append("the sync would keep a stale copy after a same-length overwrite")
        os.unlink(local)
for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
