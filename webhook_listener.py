import hashlib
import hmac
import os
import subprocess

from flask import Flask, abort, request

app = Flask(__name__)

GITHUB_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "").encode()

def verify_signature(payload, signature):
    mac = hmac.new(GITHUB_SECRET, msg=payload, digestmod=hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()
    return hmac.compare_digest(expected, signature)

@app.route("/github-webhook", methods=["POST"])
def github_webhook():
    signature = request.headers.get("X-Hub-Signature-256")
    if not signature or not verify_signature(request.data, signature):
        abort(403)
    event = request.headers.get("X-GitHub-Event")
    if event == "push":
        payload = request.json
        if payload and payload.get("ref") == "refs/heads/main":
            subprocess.Popen(
                ["/bin/bash", "/home/devuser/projects/test-rag/update_and_restart.sh"]
            )
    return "", 204

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9000)
