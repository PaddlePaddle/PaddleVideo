## sample script
# run paddlevideo server with PP-TSM:
nohup python3.7 -m paddle_serving_server.serve \
--model ./ppTSM_serving_server \
--port 9993 &

## run paddlevideo server with PP-TSN:
# nohup python3.7 -m paddle_serving_server.serve \
# --model ./ppTSN_serving_server \
# --port 9993 &
