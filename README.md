# Running as a systemd Service

To keep the Streamlit app running in the background and automatically restart it if it fails, you can set it up as a systemd service.

1. **Create the service file directly in `/etc/systemd/system`:**
   
   ```sh
   sudo vim /etc/systemd/system/streamlit.service
   ```

2. **Paste the following content (edit paths as needed):**
   
   ```ini
   [Unit]
   Description=Streamlit App
   After=network.target

   [Service]
   User=devuser
   WorkingDirectory=/home/devuser/projects/test-rag
   ExecStart=/home/devuser/projects/test-rag/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. **Enable and start the service:**
   ```sh
   sudo systemctl daemon-reload
   sudo systemctl enable streamlit
   sudo systemctl start streamlit
   ```

4. **Check the status:**
   ```sh
   sudo systemctl status streamlit
   ```

The app will now run in the background and restart automatically if it crashes. You can access it at `http://YOUR_VPS_IP:8501/`. 