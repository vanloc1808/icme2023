""" Evaluation script to evaluate Out-of-Context Detection Accuracy"""

import cv2
import os
import io
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory + "/grit/")
from utils.config import *
from utils.text_utils import get_text_metadata
from model_archs.models import CombinedModelMaskRCNN
from utils.common_utils import read_json_data
from utils.eval_utils import is_bbox_overlap, top_bbox_from_scores

#grit
# import hydra
# from omegaconf import DictConfig
from hydra import compose, initialize
from omegaconf import OmegaConf

# initialize(config_path="./grit/configs", job_name="coco_config")
# cfg = compose(config_name="config", overrides=["db=mysql", "db.user=me"])
# print(OmegaConf.to_yaml(cfg))

from grit.datasets.caption.field import TextField
from grit.datasets.caption.coco import build_coco_dataloaders
from grit.models.caption import Transformer, GridFeatureNetwork, CaptionGenerator

from grit.models.caption.detector import build_detector
from grit.models.common.attention import MemoryAttention
from grit.engine.caption_engine import *
# model
from grit.models.caption import Transformer, GridFeatureNetwork, CaptionGenerator

# dataset
from PIL import Image
from grit.datasets.caption.field import TextField
from grit.datasets.caption.transforms import get_transform
from grit.engine.utils import nested_tensor_from_tensor_list
####end grit

from transformers import DebertaTokenizer, DebertaForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import json

debert_model = pipeline("text-classification", model="microsoft/deberta-xlarge-mnli", device="cuda:0")
debert_tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
contextual_model = SentenceTransformer('sentence-transformers/stsb-bert-base')

grit_vocab=current_directory+"/grit/data/vocab.json"
grit_checkpoint=current_directory+"/grit/grit_checkpoint_4ds.pth"
GRIT_CONFIG = OmegaConf.load(current_directory+"/grit/configs/caption/coco_config.yaml")

initialize(config_path="./grit/configs/caption/", job_name="grit")
GRIT_CONFIG = compose(config_name="coco_config")
GRIT_CONFIG['exp']['checkpoint'] = grit_checkpoint
GRIT_CONFIG.dataset['vocab_path'] = grit_vocab

def build_grit_model(config):
    device = torch.device(f"cuda:0")
    detector = build_detector(config).to(device)

    grit_net = GridFeatureNetwork(
        pad_idx=config.model.pad_idx,
        d_in=config.model.grid_feat_dim,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        attention_module=MemoryAttention,
        **config.model.grit_net,
    )
    cap_generator = CaptionGenerator(
        vocab_size=config.model.vocab_size,
        max_len=config.model.max_len,
        pad_idx=config.model.pad_idx,
        cfg=config.model.cap_generator,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        **config.model.cap_generator,
    )

    model = Transformer(
        grit_net,
        cap_generator,
        detector=detector,
        use_gri_feat=config.model.use_gri_feat,
        use_reg_feat=config.model.use_reg_feat,
        config=config,
    )
    model = model.to(device)

    # load checkpoint
    model.eval()
    if os.path.exists(config.exp.checkpoint):
        checkpoint = torch.load(config.exp.checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("model missing:", len(missing))
        print("model unexpected:", len(unexpected))
        
    model.cached_features = False

    # prepare utils
    transform = get_transform(config.dataset.transform_cfg)['valid']
    text_field = TextField(vocab_path=config.vocab_path if 'vocab_path' in config else config.dataset.vocab_path)
    return model, transform, text_field

grit_model, transform, text_field =  build_grit_model(GRIT_CONFIG)

def get_grit_cap(model, transform, text_field, config, img_path):
    rgb_image = Image.open(img_path).convert('RGB')
    image = transform(rgb_image)
    images = nested_tensor_from_tensor_list([image]).to(device)
    
    # inference and decode
    with torch.no_grad():
        out, _ = model(images,                   
                    seq=None,
                    use_beam_search=True,
                    max_len=config.model.beam_len,
                    eos_idx=config.model.eos_idx,
                    beam_size=config.model.beam_size,
                    out_size=1,
                    return_probs=False,
                    )
        caption = text_field.decode(out, join_words=True)[0]
        return caption
 
context_dict = {}

# Word Embeddings
# text_field, word_embeddings, vocab_size = get_text_metadata()

# Models (create model according to text embedding)
if embed_type == 'use':
    # For USE (Universal Sentence Embeddings)
    model_name = 'img_use_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
else:
    # For Glove and Fasttext Embeddings
    model_name = 'img_lstm_glove_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(use=False, hidden_size=300, embedding_length=word_embeddings.shape[1]).to(device)


def get_scores(v_data):
    """
        Computes score for the two captions associated with the image

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            score_c1 (float): Score for the first caption associated with the image
            score_c2 (float): Score for the second caption associated with the image
    """
    checkpoint = torch.load(BASE_DIR + 'models_final/' + model_name + '.pt')
    combined_model.load_state_dict(checkpoint)
    combined_model.to(device)
    combined_model.eval()

    img_path = os.path.join(DATA_DIR, v_data["img_local_path"])
    bbox_list = v_data['maskrcnn_bboxes']
    bbox_classes = [-1] * len(bbox_list)
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    bbox_list.append([0, 0, img_shape[1], img_shape[0]])  # For entire image (global context)
    bbox_classes.append(-1)
    cap1 = v_data['caption1_modified']
    cap2 = v_data['caption2_modified']

    img_tensor = [torch.tensor(img).to(device)]
    bboxes = [torch.tensor(bbox_list).to(device)]
    bbox_classes = [torch.tensor(bbox_classes).to(device)]

    if embed_type != 'use':
        # For Glove, Fasttext embeddings
        cap1_p = text_field.preprocess(cap1)
        cap2_p = text_field.preprocess(cap2)
        embed_c1 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap1_p]).unsqueeze(
            0).to(device)
        embed_c2 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap2_p]).unsqueeze(
            0).to(device)
    else:
        # For USE embeddings
        embed_c1 = torch.tensor(use_embed([cap1]).numpy()).to(device)
        embed_c2 = torch.tensor(use_embed([cap2]).numpy()).to(device)

    with torch.no_grad():
        z_img, z_t_c1, z_t_c2 = combined_model(img_tensor, embed_c1, embed_c2, 1, [embed_c1.shape[1]],
                                               [embed_c2.shape[1]], bboxes, bbox_classes)

    z_img = z_img.permute(1, 0, 2)
    z_text_c1 = z_t_c1.unsqueeze(2)
    z_text_c2 = z_t_c2.unsqueeze(2)

    # Compute Scores
    score_c1 = torch.bmm(z_img, z_text_c1).squeeze()
    score_c2 = torch.bmm(z_img, z_text_c2).squeeze()

    return score_c1, score_c2


def evaluate_context_with_bbox_overlap(v_data):
    """
        Computes predicted out-of-context label for the given data point

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
    """
    img_path = os.path.join(DATA_DIR, v_data['img_local_path'])
#     file_name = img_path
#     file_metadata = {'name': file_name}
#     media = MediaIoBaseUpload(io.BytesIO(open(file_name, 'rb').read()), mimetype='image/jpeg')
#     file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
#     file_id = file.get('id')
#     permission = {'type': 'anyone', 'role': 'writer'}
#     drive_service.permissions().create(
#         fileId=file_id, body=permission).execute()
#     data = SearchByImageService.get_instance().search(
#         'https://drive.google.com/uc?id='+str(file_id),
#         # img_path,
#         limit_page=5
#     )
#     print("data :", data)
#     drive_service.files().delete(fileId=file_id).execute()
#     in_famous = check_famous(data)
    
    grit_cap = get_grit_cap(grit_model, transform, text_field, GRIT_CONFIG, img_path)

    bboxes = v_data['maskrcnn_bboxes']
    score_c1, score_c2 = get_scores(v_data)
    textual_sim = float(v_data['bert_base_score'])

    process_embedding1 = v_data['caption1']
    process_embedding2 = v_data['caption2']

    debert_sentence_1 = process_embedding1 + process_embedding2
    debert_sentence_2 = process_embedding2 + process_embedding1

    nli_score_1 = debert_model(debert_sentence_1)[0]['label']
    nli_score_2 = debert_model(debert_sentence_2)[0]['label']

    top_bbox_c1 = top_bbox_from_scores(bboxes, score_c1)
    top_bbox_c2 = top_bbox_from_scores(bboxes, score_c2)
    bbox_overlap = is_bbox_overlap(top_bbox_c1, top_bbox_c2, iou_overlap_threshold)
    
    embeddings_img_cap_grit = contextual_model.encode(grit_cap, convert_to_tensor=True)
    
    embeddings1 = contextual_model.encode(process_embedding1, convert_to_tensor=True)
    embeddings2 = contextual_model.encode(process_embedding2, convert_to_tensor=True)
    
    cosine_scores1_grit = util.cos_sim(embeddings1, embeddings_img_cap_grit)
    cosine_scores2_grit = util.cos_sim(embeddings2, embeddings_img_cap_grit)
    emds_sim = util.cos_sim(embeddings1, embeddings2)
    
    IC_NER_GRIT = ((cosine_scores1_grit > 0.5 and len(v_data['caption1_entities']) < 1) \
                or (cosine_scores2_grit > 0.5 and len(v_data['caption2_entities']) < 1))
    
    in_our = 0
    in_them = 0

    con1 = (nli_score_1 == "ENTAILMENT" and nli_score_2 != "CONTRADICTION")
    con2 = (nli_score_2 == "ENTAILMENT" and nli_score_1 != "CONTRADICTION")

    if nli_score_1 == "CONTRADICTION" and nli_score_2 == "CONTRADICTION" and textual_sim >= 0.25:
        context = 1
        cosmos_context = 1
    elif con1 or con2:
            context = 0
            cosmos_context=0
    elif bbox_overlap:

    # if bbox_overlap:
        # Check for captions with same context : Same grounding with high textual overlap (Not out of context)
        if textual_sim >= textual_sim_threshold:
            cosmos_context = 0
        # Check for captions with different context : Same grounding with low textual overlap (Out of context)
        else:
                cosmos_context = 1
        
        if emds_sim >= textual_sim_threshold:
            context = 0
        # Check for captions with different context : Same grounding with low textual overlap (Out of context)
        else:
                context = 1
    else:
        # Check for captions with same context : Different grounding (Not out of context)
        return 0, 0
    return  context, cosmos_context


if __name__ == "__main__":
    """ Main function to compute out-of-context detection accuracy"""

    test_samples = read_json_data(os.path.join(DATA_DIR, 'test.json'))
    cosmos_correct = 0
    ours_correct = 0
    lang_correct = 0

    for i, v_data in tqdm(enumerate(test_samples)):
        actual_context = int(v_data['context_label'])
        language_context = 0 if float(v_data['bert_base_score']) >= textual_sim_threshold else 1
        pred_context_ours, pred_context_cosmos = evaluate_context_with_bbox_overlap(v_data)

        if pred_context_ours == actual_context:
            ours_correct += 1
        if pred_context_cosmos == actual_context:
            cosmos_correct += 1

        if language_context == actual_context:
            lang_correct += 1

    print("Accuracy", cosmos_correct / len(test_samples))
