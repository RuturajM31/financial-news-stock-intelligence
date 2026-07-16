from pathlib import Path
import pytest
from financial_news_intelligence.models.full_bert_inference import DEFAULT_REMOTE_MODEL_REPO, LABEL_ORDER, resolve_model_directory, validate_model_snapshot

FILES=("config.json","model.safetensors","tokenizer.json","tokenizer_config.json","vocab.txt")
def complete(path: Path):
    path.mkdir(parents=True,exist_ok=True)
    for name in FILES: (path/name).write_text("x",encoding="utf-8")

def test_complete_local_model_has_priority(tmp_path):
    local=tmp_path/"local";complete(local)
    assert resolve_model_directory(local,environ={},secrets={},snapshot_download=lambda **_: pytest.fail("remote called"))==local.resolve()

def test_missing_local_uses_default_remote_and_env_token(tmp_path):
    remote=tmp_path/"remote";complete(remote);seen={}
    def download(**kwargs): seen.update(kwargs);return str(remote)
    assert resolve_model_directory(tmp_path/"missing",environ={"HF_TOKEN":"secret"},secrets={},snapshot_download=download)==remote.resolve()
    assert seen["repo_id"]==DEFAULT_REMOTE_MODEL_REPO and seen["token"]=="secret"

def test_environment_repo_override_wins(tmp_path):
    remote=tmp_path/"remote";complete(remote);seen={}
    resolve_model_directory(tmp_path/"missing",environ={"HF_TOKEN":"secret","HF_MODEL_REPO":"owner/model"},secrets={},snapshot_download=lambda **kw:(seen.update(kw) or str(remote)))
    assert seen["repo_id"]=="owner/model"

def test_streamlit_secret_repo_and_token_are_supported(tmp_path):
    remote=tmp_path/"remote";complete(remote);seen={}
    resolve_model_directory(tmp_path/"missing",environ={},secrets={"HF_TOKEN":"private","HF_MODEL_REPO":"secret/model"},snapshot_download=lambda **kw:(seen.update(kw) or str(remote)))
    assert seen["repo_id"]=="secret/model" and seen["token"]=="private"

def test_missing_token_fails_without_fallback(tmp_path):
    with pytest.raises(RuntimeError,match="HF_TOKEN"):
        resolve_model_directory(tmp_path/"missing",environ={},secrets={})

def test_remote_errors_are_sanitized(tmp_path):
    token="hf_super_secret"
    def fail(**_): raise RuntimeError(token)
    with pytest.raises(RuntimeError) as error:
        resolve_model_directory(tmp_path/"missing",environ={"HF_TOKEN":token},secrets={},snapshot_download=fail)
    assert token not in str(error.value)

def test_incomplete_snapshot_is_rejected(tmp_path):
    remote=tmp_path/"remote";remote.mkdir();(remote/"config.json").write_text("x")
    with pytest.raises(RuntimeError,match="incomplete"):
        resolve_model_directory(tmp_path/"missing",environ={"HF_TOKEN":"secret"},secrets={},snapshot_download=lambda **_:str(remote))

def test_weight_bin_is_accepted_and_class_order_stays_fixed(tmp_path):
    path=tmp_path/"model";complete(path);(path/"model.safetensors").unlink();(path/"pytorch_model.bin").write_text("x")
    assert not validate_model_snapshot(path)
    assert LABEL_ORDER==("Bearish","Neutral","Bullish")