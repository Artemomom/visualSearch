#!/bin/bash
set -eu

envsubst '${API_BASE_URL} ${APP_VERSION} ${APP_NAME}' </etc/nginx/conf.d/default.conf.template >/etc/nginx/conf.d/default.conf

exec "$@"
