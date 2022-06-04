unset GREP_OPTIONS

function install_trt(){
  CUDA_VERSION=$(nvcc --version | egrep -o "V[0-9]+.[0-9]+" | cut -c2-)
  if [ $CUDA_VERSION == "10.2" ]; then
    wget https://paddle-ci.gz.bcebos.com/TRT/TensorRT6-cuda10.2-cudnn7.tar.gz --no-check-certificate
    tar -zxf TensorRT6-cuda10.2-cudnn7.tar.gz -C /usr/local
    cp -rf /usr/local/TensorRT-6.0.1.8/include/*  /usr/include/ && cp -rf /usr/local/TensorRT-6.0.1.8/lib/* /usr/lib/
    rm -rf TensorRT6-cuda10.2-cudnn7.tar.gz
  elif [ $CUDA_VERSION == "11.2" ]; then
    wget https://paddle-ci.gz.bcebos.com/TRT/TensorRT-8.0.3.4.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz --no-check-certificate
    tar -zxf TensorRT-8.0.3.4.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz -C /usr/local
    cp -rf /usr/local/TensorRT-8.0.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.0.3.4/lib/* /usr/lib/
    rm -rf TensorRT-8.0.3.4.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz
  else
    echo "No Cuda Found, no need to install TensorRT"
  fi
}

function env_install()
{
    apt install -y libcurl4-openssl-dev libbz2-dev
    wget https://paddle-serving.bj.bcebos.com/others/centos_ssl.tar && tar xf centos_ssl.tar && rm -rf centos_ssl.tar && mv libcrypto.so.1.0.2k /usr/lib/libcrypto.so.1.0.2k && mv libssl.so.1.0.2k /usr/lib/libssl.so.1.0.2k && ln -sf /usr/lib/libcrypto.so.1.0.2k /usr/lib/libcrypto.so.10 && ln -sf /usr/lib/libssl.so.1.0.2k /usr/lib/libssl.so.10 && ln -sf /usr/lib/libcrypto.so.10 /usr/lib/libcrypto.so && ln -sf /usr/lib/libssl.so.10 /usr/lib/libssl.so
    rm -rf /usr/local/go && wget -qO- https://paddle-ci.gz.bcebos.com/go1.15.12.linux-amd64.tar.gz | \
    tar -xz -C /usr/local && \
    mkdir /root/go && \
    mkdir /root/go/bin && \
    mkdir /root/go/src && \
    echo "GOROOT=/usr/local/go" >> /root/.bashrc && \
    echo "GOPATH=/root/go" >> /root/.bashrc && \
    echo "PATH=/usr/local/go/bin:/root/go/bin:$PATH" >> /root/.bashrc
    install_trt
}

env_install
