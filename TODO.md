1. context-aware
2. regex match
3. 是否有实体重叠问题，如“张某的小电驴”：是，目前召回率较高、精确率较低
4. R-Drop
5. VAT
6. FGM
7. 分析三类错误：真实未预测出的、预测出的边界错误、预测出的类别错误
8. K折：是，已添加
9. hfl/chinese-legal-electra-base-discriminator
    ``` py
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    model_name_or_path = "hfl/chinese-legal-electra-base-discriminator"
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    ```
10. 改题实体普遍较长，试一下bert_pointer：分类准确，不是模型问题
11. 分析数据，后处理需要保留的实体