FROM cr.msk.sbercloud.ru/aicloud-base-images/horovod-cuda11.5-pt1.11.0-a100:0.0.32.3

USER root
WORKDIR /app

RUN pip3 install cmake --upgrade

RUN wget https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.22.1-Source.tar.gz \
    && tar -xzf eccodes-2.22.1-Source.tar.gz \
    && mkdir build ; cd build ; cmake -DCMAKE_INSTALL_PREFIX=/usr ../eccodes-2.22.1-Source \
    && cd /app/build ; make -s -j 4; ctest ; make install

RUN apt update
RUN apt-get install -y espeak git-lfs
COPY . .
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN git lfs clone https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft
USER user

