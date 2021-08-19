FROM python:3.8.7-buster
RUN apt-get update -y
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
#RUN curl -o pydust_core-2.0.0-cp38-cp38-linux_x86_64.whl https://dai.idlab.uantwerpen.be/dust/downloads-index/DUST_Dust20Python_LinuxX8664Python38/pydust_core-2.0.0-cp38-cp38-linux_x86_64.whl
#RUN curl -o dust_2.0.0_amd64.deb https://dai.idlab.uantwerpen.be/dust/downloads-index/DUST_DustCore_LinuxX8664/dust_2.0.0_amd64.deb4


RUN apt install -y git
RUN apt install -y cmake
RUN git clone https://github.com/eclipse/mosquitto.git && \

        cd mosquitto && \

        git checkout v1.6.9 && \

        mkdir build && \

        cd build && \

        cmake -DCMAKE_BUILD_TYPE=Release -DWITH_BUNDLED_DEPS=ON -DDOCUMENTATION=OFF -DWITH_PIC=ON -DWITH_STATIC_LIBRARIES=ON .. && \

        make -j4 install && \

        cd ../.. && \

        rm -rfd mosquitto



COPY /app .
RUN ls
RUN dpkg -i ./dust_2.0.0_amd64.deb
RUN pip3 install ./pydust_core-2.0.0-cp38-cp38-linux_x86_64.whl
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3","-u","./seg_inference.py"]
