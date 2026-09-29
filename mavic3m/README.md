# Mavic 3M Real-time Video Streaming

Este projeto configura um servidor de vídeo e roda um script Python para processar o fluxo ao vivo (RTMP) do DJI Mavic 3M usando YOLO (Ultralytics) e OpenCV.

## 1. Instalar e rodar o Servidor de Vídeo (MediaMTX)

O MediaMTX é um servidor RTMP simples que vai receber o vídeo do rádio do seu drone. Manter a janela aberta enquanto estiver recebendo o vídeo.

Install: https://github.com/bluenviron/mediamtx/releases

Run mediamtx.exe

## 2. Configurar o Drone (DJI Pilot 2)

Com o servidor rodando no seu computador:
1. Conecte o rádio do Mavic 3M na mesma rede de internet que o computador.
2. Descubra o IP do computador na rede (rodar`ipconfig` no terminal).
3. No DJI Pilot 2: **Configurações > Transmissão > Plataforma de Transmissão ao Vivo > RTMP**.
4. Digite o endereço: `rtmp://IP:1935/live/drone` (`IP`: IP do PC).
5. Inicie a transmissão no aplicativo.

## 3. Run

O script vai abrir o stream `rtmp://localhost:1935/live/drone` e abrirá uma janela mostrando o vídeo em tempo real processado com a inteligência artificial (YOLO).

```powershell
   uv venv
   uv pip install -r requirements.txt
   uv run main.py
```
