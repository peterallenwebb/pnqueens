#!/usr/bin/env python3
"""Google Cloud Storage with nothing but the standard library, openssl and
curl: V4 signed URLs made from a service-account key, and an access token
for the JSON API (bucket settings, listing and downloads).

The key file (the JSON a service account key is downloaded as) stays on the
operator's machine; only the URLs it signs travel to the boxes. Each signed
URL allows one method on one object for at most seven days.
"""
import base64, datetime, hashlib, json, os, subprocess, tempfile, time, urllib.parse, urllib.request

STORAGE = "https://storage.googleapis.com"


def load_key(path):
    with open(path) as f:
        return json.load(f)


def _rsa_sha256(pem, data):
    """RSA-SHA256 signature of `data` with the PEM private key, via openssl
    (the key is written to a private temporary file for the call)."""
    fd, path = tempfile.mkstemp()          # created 0600
    try:
        os.write(fd, pem.encode())
        os.close(fd)
        return subprocess.run(["openssl", "dgst", "-sha256", "-sign", path], input=data, capture_output=True, check=True).stdout
    finally:
        os.unlink(path)


def sign_url(key, bucket, obj, method="PUT", expires=7 * 86400):
    """A V4 signed URL for `method` on gs://bucket/obj, valid `expires` seconds."""
    if not 1 <= expires <= 604800:
        raise ValueError("a V4 signed URL lasts at most 7 days")
    now = datetime.datetime.now(datetime.timezone.utc)
    date, stamp = now.strftime("%Y%m%d"), now.strftime("%Y%m%dT%H%M%SZ")
    scope = f"{date}/auto/storage/goog4_request"
    path = f"/{bucket}/" + urllib.parse.quote(obj, safe="/")
    query = {
        "X-Goog-Algorithm": "GOOG4-RSA-SHA256",
        "X-Goog-Credential": f"{key['client_email']}/{scope}",
        "X-Goog-Date": stamp,
        "X-Goog-Expires": str(expires),
        "X-Goog-SignedHeaders": "host",
    }
    canonical_query = "&".join(f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}" for k, v in sorted(query.items()))
    canonical_request = "\n".join([method, path, canonical_query, "host:storage.googleapis.com\n", "host", "UNSIGNED-PAYLOAD"])
    string_to_sign = "\n".join(["GOOG4-RSA-SHA256", stamp, scope, hashlib.sha256(canonical_request.encode()).hexdigest()])
    signature = _rsa_sha256(key["private_key"], string_to_sign.encode()).hex()
    return f"{STORAGE}{path}?{canonical_query}&X-Goog-Signature={signature}"


def _b64url(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def access_token(key, scope="https://www.googleapis.com/auth/devstorage.read_only"):
    """An OAuth access token for the service account (a signed JWT exchanged at the token endpoint)."""
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    claims = _b64url(json.dumps({"iss": key["client_email"], "scope": scope, "aud": key["token_uri"], "iat": now, "exp": now + 3600}).encode())
    signing_input = f"{header}.{claims}".encode()
    jwt = signing_input.decode() + "." + _b64url(_rsa_sha256(key["private_key"], signing_input))
    data = urllib.parse.urlencode({"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": jwt}).encode()
    with urllib.request.urlopen(urllib.request.Request(key["token_uri"], data=data)) as resp:
        return json.load(resp)["access_token"]


def api_get(token, url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})) as resp:
        return resp.read()


def bucket_settings(token, bucket):
    return json.loads(api_get(token, f"{STORAGE}/storage/v1/b/{bucket}?fields=versioning,iamConfiguration,location"))


def list_objects(token, bucket, prefix):
    """The names and sizes of the objects under `prefix`."""
    out, page = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": "1000"}
        if page:
            q["pageToken"] = page
        data = json.loads(api_get(token, f"{STORAGE}/storage/v1/b/{bucket}/o?" + urllib.parse.urlencode(q)))
        out += [(o["name"], int(o["size"])) for o in data.get("items", [])]
        page = data.get("nextPageToken")
        if not page:
            return out


def download(token, bucket, obj, path):
    data = api_get(token, f"{STORAGE}/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media")
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return len(data)


def curl_put(url, path):
    """Uploads `path` to a signed PUT URL; returns (ok, message)."""
    r = subprocess.run(["curl", "-sS", "-f", "--max-time", "300", "-o", "/dev/null", "-X", "PUT", "-T", path, url], capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or "").strip()
