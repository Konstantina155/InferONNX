import os
import sys
import subprocess
from huggingface_hub import login
from huggingface_hub import hf_hub_download
from optimum.exporters.onnx import main_export
from transformers import GPT2Tokenizer, LlamaTokenizerFast

# Okay with qwen and mistral

login(token="your_hf_token_here")

def format_model_dir(output_dir, model_name_str):
    subprocess.run(f"find '{output_dir}' -maxdepth 1 -type f ! -name 'tokenizer.json' ! -name 'model.onnx' -delete", shell=True, check=True)
    subprocess.run(f"mv {output_dir}/model.onnx {output_dir}/{model_name_str}.onnx && mkdir {output_dir}/test_data_set_0 && mv {output_dir}/tokenizer.json {output_dir}/test_data_set_0/", shell=True, check=True)

def export_model(model_path):
    # 1) GPT2-124M
    # model_name = "openai-community/gpt2"
    # gpt2_output_dir = f"{model_path}/gpt"
    # os.makedirs(gpt2_output_dir, exist_ok=True)
    # main_export(
    #     model_name_or_path=model_name,
    #     output=gpt2_output_dir,
    #     task="text-generation",
    #     framework="pt"
    # )
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # tokenizer.save_pretrained(gpt2_output_dir)
    # format_model_dir(gpt2_output_dir, "gpt")

    # 2) Smol-llama-220M
    model_name = "BEE-spoke-data/smol_llama-220M-GQA"
    smol_llama_output_dir = f"{model_path}/llama"
    os.makedirs(smol_llama_output_dir, exist_ok=True)
    main_export(
        model_name_or_path=model_name,
        output=smol_llama_output_dir,
        task="text-generation",
        library="transformers",
        use_past=False,
        trust_remote_code=True
    )
    format_model_dir(smol_llama_output_dir, "llama")

    # 3) Mistral-300M
    model_name = "yuiseki/YuisekinAI-mistral-0.3B"
    mistral_output_dir = f"{model_path}/mistral"
    main_export(
        model_name_or_path=model_name,
        output=mistral_output_dir,
        task="text-generation",
        framework="pt",
        library="transformers",
        trust_remote_code=True
    )
    tokenizer_path = f"{model_path}/mistral/tokenizer.model"
    try:
        tokenizer = LlamaTokenizerFast(vocab_file=tokenizer_path, from_slow=True)
    except Exception as e:
        print("Error: ", e)
        exit(1)
    tokenizer.save_pretrained(mistral_output_dir)
    format_model_dir(mistral_output_dir, "mistral")

    # 4) Qwen2.5-0.5B
    model_name = "Qwen/Qwen2.5-0.5B"
    qwen_output_dir = f"{model_path}/qwen"
    os.makedirs(qwen_output_dir, exist_ok=True)
    main_export(
        model_name_or_path=model_name,
        output=qwen_output_dir,
        task="text-generation",
        framework="pt",
    )
    format_model_dir(qwen_output_dir, "qwen")

    # qwen2.5-0.5B's original tokenizer is not working in Tract inference engine
    subprocess.run(f"rm -f {qwen_output_dir}/test_data_set_0/tokenizer.json", shell=True, check=True)
    local_file_path = hf_hub_download(
        repo_id="onnx-community/Qwen2.5-0.5B",
        filename="tokenizer.json",
        local_dir=f"{qwen_output_dir}/test_data_set_0/" 
    )

def below():
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer

    # This is the industry standard NER model
    model_id = "dslim/bert-base-NER"
    save_dir = "./clean_bert_ner"

    print("Downloading and exporting model to ONNX...")
    model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"\nSUCCESS! Your files are in the '{save_dir}' folder.")
    print(f"Model path: {save_dir}/model.onnx")
    print(f"Tokenizer path: {save_dir}/tokenizer.json")

def main():
    if len(sys.argv) != 2:
        print("python3 download_llms.py <inferONNX_path>")
        sys.exit(1)

    inferONNX_path = sys.argv[1]
    model_path = inferONNX_path + "models/"
    export_model(model_path)

if __name__ == "__main__":
    main()