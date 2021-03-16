FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y locales git libsndfile1 \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN mkdir -p /work
RUN mkdir -p /data
RUN mkdir -p /tmp
WORKDIR /work

ADD requirements.txt /work

RUN pip install --ignore-installed --no-cache -r requirements.txt

ADD rpspp /work/rpspp
