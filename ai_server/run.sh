# prod
CUDA_VISIBLE_DEVICES=0 gunicorn -c gunicorn_conf_llm_server_base.py manage:app
