# predict.py
import torch
from typing import List, Tuple
import json

from HW3.config import Config
from HW3.model.net import BiLSTMCRF


class NERPredictor:
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device if device is not None else config.device
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        对输入文本进行NER预测

        Args:
            text: 输入文本

        Returns:
            列表，每个元素为(实体文本, 实体类型, 起始位置, 结束位置)
        """
        # 字符转换为ID
        char_ids = [self.config.char2idx.get(char, self.config.char2idx['<UNK>'])
                    for char in text]

        # 转换为tensor
        input_tensor = torch.tensor([char_ids]).to(self.device)
        mask = torch.ones(input_tensor.shape, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            emissions = self.model(input_tensor, mask)
            pred_tags = self.model.crf.decode(emissions, mask=mask)[0]

        # 将预测标签转换为实体
        entities = []
        current_entity = None

        for i, tag_id in enumerate(pred_tags):
            tag = self.config.idx2tag[tag_id]
            if tag.startswith('B-'):
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {
                    'text': text[i],
                    'type': tag[2:],
                    'start': i,
                    'end': i + 1
                }
            elif tag.startswith('I-') and current_entity is not None:
                if tag[2:] == current_entity['type']:
                    current_entity['text'] += text[i]
                    current_entity['end'] = i + 1
                else:
                    entities.append(current_entity)
                    current_entity = None
            elif tag == 'O':
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity is not None:
            entities.append(current_entity)

        # 转换为输出格式
        return [(e['text'], e['type'], e['start'], e['end']) for e in entities]

    def predict_batch(self, texts: List[str]) -> List[List[Tuple[str, str, int, int]]]:
        """批量预测"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def format_result(text: str, entities: List[Tuple[str, str, int, int]]) -> str:
    """格式化输出结果"""
    result = list(text)
    for entity, type_, start, end in reversed(entities):
        result.insert(end, f"</{type_}>")
        result.insert(start, f"<{type_}>")
    return ''.join(result)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NER Prediction')
    parser.add_argument('--model_path', required=True, help='Path to the model checkpoint')
    parser.add_argument('--config_path', required=True, help='Path to the configuration file')
    parser.add_argument('--input_file', required=True, help='Input file path')
    parser.add_argument('--output_file', required=True, help='Output file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    args = parser.parse_args()

    # 加载配置
    with open(args.config_path, 'r') as f:
        config = Config(**json.load(f))

    # 加载模型
    model = BiLSTMCRF(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_tags=config.num_tags,
        dropout=0.0  # 预测时不需要dropout
    )
    model.load_state_dict(torch.load(args.model_path))

    # 创建预测器
    predictor = NERPredictor(model, config)

    # 读取输入文本
    with open(args.input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    # 批量预测
    results = []
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i + args.batch_size]
        batch_results = predictor.predict_batch(batch_texts)
        results.extend(batch_results)

    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for text, entities in zip(texts, results):
            formatted = format_result(text, entities)
            f.write(formatted + '\n')


if __name__ == '__main__':
    main()