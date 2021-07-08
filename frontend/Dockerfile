FROM node:14.11.0 as build

WORKDIR /app
ENV PATH /app/node_module/.bin:$PATH

ADD package.json yarn.lock /tmp/
RUN cd /tmp && yarn
RUN cd /app && ln -s /tmp/node_modules

ARG REACT_APP_API_URL="https://vs.griddynamics.net/fashion-search-deit/api"
ARG REACT_APP_ROUTE_URL="/fashion-search-deit"

COPY . ./

RUN printf "REACT_APP_API_URL=${REACT_APP_API_URL}\nREACT_APP_ROUTE_URL=${REACT_APP_ROUTE_URL}" > .env && \
    sed -i -E "s@\"homepage\":.*@\"homepage\": \"${REACT_APP_ROUTE_URL}\"@" package.json

RUN yarn build --network-timeout 100000

FROM nginxinc/nginx-unprivileged:latest
COPY --from=build /app/build /usr/share/nginx/html
COPY config/nginx/nginx.conf /etc/nginx/conf.d/default.conf.template

COPY ./entrypoint.sh entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
