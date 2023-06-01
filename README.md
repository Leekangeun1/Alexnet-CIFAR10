# Alexnet-CIFAR10

---

> Cuda 가속화 방법
> 
1. NVIDIA Cuda Toolkit 설치
- https://developer.nvidia.com/cuda-toolkit-archive

2. CuDNN 설치
- 같은 사이트에서 회원가입/로그인 후 CuDNN 설치 탭에서 zip 파일 다운로드
- 다운로드 후 파일을 Cuda Toolkit 디렉토리에 복사-붙여넣기

3. (optional) 환경변수에서 Cuda의 경로가 제대로 들어갔는지 확인

4. IDE(ex. 파이참)에서 torch, torchvision 등 버전에 맞게 재설치
- 만약 이미 설치되어 있다면, uninstall
- 그리고 pyTorch 사이트에서 설정환경에 알맞은 버전으로 설치
- 터미널에서 pip 명령어로 재설치할 수 있다.
- [https://nanunzoey.tistory.com/entry/torchcudaisavailable-False-해결](https://nanunzoey.tistory.com/entry/torchcudaisavailable-False-%ED%95%B4%EA%B2%B0)
