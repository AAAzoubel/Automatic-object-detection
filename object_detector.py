# Importa a classe YOLO da biblioteca Ultralytics.
# É ela que carrega o modelo treinado e faz detecção/tracking nos frames.
from ultralytics import YOLO

# OpenCV:
# usado para abrir o vídeo, ler frame por frame, redimensionar imagens
# e também salvar o vídeo final anotado.
import cv2

# Pandas:
# usado para transformar a lista de detecções em tabela
# e exportar para CSV / Excel.
import pandas as pd

# OS:
# usado para criar pastas automaticamente e lidar com caminhos.
import os


# --------------------------------------------------
# Carrega o modelo YOLO
# --------------------------------------------------
def load_model():
    # Aqui carregamos o arquivo do modelo pré-treinado.
    # "yolov8n.pt" é uma versão leve e rápida do YOLOv8.
    return YOLO("yolov8n.pt")


# --------------------------------------------------
# Processa um único frame do vídeo
# --------------------------------------------------
def process_frame(model, frame, frame_number):
    # model.track faz detecção + tracking.
    # Tracking significa rastrear um objeto pra que não seja classificado como diferente a cada frame
    #
    # persist=True ajuda o modelo a "lembrar" os objetos entre os frames,
    # em vez de tratar tudo como novo a cada imagem.
    tracking_results = model.track(frame, persist=True)

    # tracking_results é uma lista de resultados.
    # Como estamos processando um frame por vez, pegamos o primeiro.
    frame_result = tracking_results[0]

    # Lista que vai armazenar todas as detecções encontradas
    # SOMENTE neste frame atual.
    detections_in_frame = []

    # frame_result.boxes contém as bounding boxes detectadas no frame.
    # Se não houver detecção, frame_result.boxes pode vir vazio.
    if frame_result.boxes is not None:

        # Percorre cada box detectada nesse frame.
        for box in frame_result.boxes:

            # box.cls guarda o ID numérico da classe detectada.
            # Exemplo: 0 pode ser "person", 2 pode ser "car", etc.
            class_id = int(box.cls[0])

            # model.names converte o ID numérico em nome da classe.
            # Exemplo: 0 -> "person"
            class_name = model.names[class_id]

            # box.conf guarda a confiança da detecção.
            # Quanto mais próximo de 1, maior a confiança do modelo.
            confidence = float(box.conf[0])

            # box.xyxy guarda as coordenadas da bounding box:
            # x1, y1 -> canto superior esquerdo
            # x2, y2 -> canto inferior direito
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # track_id é o ID do objeto rastreado.
            # Se o tracking conseguir acompanhar o objeto ao longo do vídeo,
            # esse ID tende a permanecer o mesmo nos frames seguintes.
            track_id = None

            # Nem sempre o YOLO consegue gerar ID em todos os casos.
            # Por isso fazemos a verificação antes de converter.
            if box.id is not None:
                track_id = int(box.id[0])

            # Guardamos os dados estruturados dessa detecção.
            # Isso será depois exportado para CSV/Excel e usado em estatísticas.
            detections_in_frame.append({
                "frame": frame_number,   # número do frame em que o objeto apareceu
                "id": track_id,          # ID rastreado do objeto
                "classe": class_name,    # nome da classe detectada
                "confianca": confidence, # confiança do modelo
                "x1": x1,                # coordenada x inicial
                "y1": y1,                # coordenada y inicial
                "x2": x2,                # coordenada x final
                "y2": y2                 # coordenada y final
            })

    # frame_result.plot() desenha no frame:
    # - bounding boxes
    # - nome da classe
    # - confiança
    # - tracking ID (quando disponível)
    #
    # Esse frame anotado será usado para gerar o vídeo final.
    annotated_frame = frame_result.plot()

    # Retornamos:
    # 1. a lista de detecções estruturadas do frame
    # 2. o frame já desenhado/anotado
    return detections_in_frame, annotated_frame


# --------------------------------------------------
# Gera estatísticas a partir do DataFrame de detecções
# --------------------------------------------------
def generate_statistics(detections_df):
    # Garante que a pasta outputs exista antes de salvar arquivos nela.
    os.makedirs("outputs", exist_ok=True)

    # Se o DataFrame estiver vazio, significa que nada foi detectado.
    # Então não faz sentido gerar estatísticas.
    if detections_df.empty:
        return {
            "stats_total_path": None,
            "stats_unique_path": None
        }

    # ------------------------------------------
    # Estatística 1: total de detecções por classe
    # ------------------------------------------
    # value_counts() conta quantas vezes cada classe apareceu.
    # Exemplo:
    # person -> 300
    # car -> 50
    detections_per_class = detections_df["classe"].value_counts().reset_index()

    # Renomeando as colunas para ficar mais claro no arquivo final.
    detections_per_class.columns = ["classe", "quantidade_deteccoes"]

    # Caminho do arquivo de estatísticas de detecções totais.
    stats_total_path = "outputs/estatisticas_deteccoes.csv"

    # Salva o CSV sem a coluna de índice.
    detections_per_class.to_csv(stats_total_path, index=False)

    # ------------------------------------------
    # Estatística 2: quantidade de objetos únicos por classe
    # ------------------------------------------
    # Aqui queremos saber quantos IDs diferentes apareceram por classe.
    # Para isso, primeiro removemos as linhas que não possuem ID.
    tracked_df = detections_df.dropna(subset=["id"]).copy()

    # Se ainda houver linhas com ID, conseguimos calcular objetos únicos.
    if not tracked_df.empty:

        # Convertendo a coluna "id" para inteiro.
        # Isso evita problemas caso venha como float por causa do pandas.
        tracked_df["id"] = tracked_df["id"].astype(int)

        # groupby("classe") agrupa por classe.
        # nunique() conta quantos IDs distintos existem em cada grupo.
        #
        # Exemplo:
        # person -> 5 IDs únicos
        # car -> 2 IDs únicos
        unique_objects_per_class = (
            tracked_df.groupby("classe")["id"]
            .nunique()
            .reset_index()
        )

        # Renomeando as colunas para melhor entendimento.
        unique_objects_per_class.columns = ["classe", "quantidade_objetos_unicos"]

        # Caminho do arquivo de objetos únicos.
        stats_unique_path = "outputs/estatisticas_objetos_unicos.csv"

        # Salva o CSV.
        unique_objects_per_class.to_csv(stats_unique_path, index=False)

    else:
        # Se não houver nenhum ID rastreado, não há estatística de objetos únicos.
        stats_unique_path = None

    # Retorna os caminhos dos arquivos gerados.
    return {
        "stats_total_path": stats_total_path,
        "stats_unique_path": stats_unique_path
    }


# --------------------------------------------------
# Exporta todas as detecções para arquivos
# --------------------------------------------------
def export_results(all_detections):
    # Garante que a pasta outputs exista.
    os.makedirs("outputs", exist_ok=True)

    # Converte a lista de dicionários em DataFrame.
    # Isso facilita exportar e analisar.
    detections_df = pd.DataFrame(all_detections)

    # Caminhos dos arquivos principais.
    excel_path = "outputs/deteccoes_video.xlsx"
    csv_path = "outputs/deteccoes_video.csv"

    # Exporta para Excel.
    detections_df.to_excel(excel_path, index=False)

    # Exporta para CSV.
    detections_df.to_csv(csv_path, index=False)

    # Gera estatísticas complementares a partir do DataFrame.
    stats_info = generate_statistics(detections_df)

    # Retorna todos os caminhos dos arquivos gerados.
    return {
        "excel_path": excel_path,
        "csv_path": csv_path,
        **stats_info
    }


# --------------------------------------------------
# Monta um resumo em JSON para a API retornar
# --------------------------------------------------
def build_summary(all_detections):
    # Converte a lista de detecções em DataFrame.
    detections_df = pd.DataFrame(all_detections)

    # Se estiver vazio, devolve um resumo vazio.
    if detections_df.empty:
        return {
            "total_detections": 0,
            "classes_detected": [],
            "detections_by_class": {},
            "unique_objects_by_class": {}
        }

    # Conta quantas detecções houve por classe.
    detections_by_class = detections_df["classe"].value_counts().to_dict()

    # Lista de classes detectadas.
    classes_detected = list(detections_by_class.keys())

    # Filtra apenas linhas com ID válido.
    tracked_df = detections_df.dropna(subset=["id"]).copy()

    # Calcula quantidade de objetos únicos por classe.
    if not tracked_df.empty:
        tracked_df["id"] = tracked_df["id"].astype(int)

        unique_objects_by_class = (
            tracked_df.groupby("classe")["id"]
            .nunique()
            .to_dict()
        )
    else:
        unique_objects_by_class = {}

    # Retorna um resumo pronto para a API devolver em JSON.
    return {
        "total_detections": len(detections_df),
        "classes_detected": classes_detected,
        "detections_by_class": detections_by_class,
        "unique_objects_by_class": unique_objects_by_class
    }


# --------------------------------------------------
# Processa o vídeo inteiro
# --------------------------------------------------
def process_video(video_path, model):
    # Garante que a pasta outputs exista.
    os.makedirs("outputs", exist_ok=True)

    # Abre o vídeo para leitura.
    video_capture = cv2.VideoCapture(video_path)

    # Codec usado para salvar o vídeo final.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Caminho do vídeo de saída.
    output_video_path = "outputs/video_detectado.mp4"

    # Cria o objeto que vai salvar o vídeo anotado.
    # Parâmetros:
    # - caminho de saída
    # - codec
    # - FPS (aqui fixado em 30)
    # - resolução do frame
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (960, 540))

    # Lista com todas as detecções de todos os frames.
    all_detections = []

    # Contador de frame.
    frame_number = 0

    # Loop principal: continua até acabar o vídeo.
    while True:
        # ret indica se o frame foi lido com sucesso.
        # frame é a imagem em si.
        ret, frame = video_capture.read()

        # Se não conseguiu ler, o vídeo terminou ou houve erro.
        if not ret:
            break

        # Incrementa o contador de frames.
        frame_number += 1

        # Redimensiona todos os frames para um tamanho fixo.
        # Isso padroniza a entrada e reduz custo computacional.
        frame = cv2.resize(frame, (960, 540))

        # Processa o frame atual:
        # - detecta/rastreia objetos
        # - gera os dados estruturados
        # - gera o frame anotado
        detections_in_frame, annotated_frame = process_frame(model, frame, frame_number)

        # Junta as detecções desse frame à lista geral.
        all_detections.extend(detections_in_frame)

        # Escreve o frame anotado no vídeo de saída.
        video_writer.write(annotated_frame)

    # Libera o vídeo de entrada da memória.
    video_capture.release()

    # Finaliza o arquivo do vídeo de saída.
    video_writer.release()

    # Exporta CSV, Excel e estatísticas.
    exported_files = export_results(all_detections)

    # Gera o resumo que a API vai retornar.
    summary = build_summary(all_detections)

    # Retorna tudo que a API pode precisar.
    return {
        "detections": all_detections,               # lista completa de detecções
        "summary": summary,                         # resumo em JSON
        "output_video_path": output_video_path,     # caminho do vídeo anotado
        **exported_files                            # caminhos de csv/excel/estatísticas
    }