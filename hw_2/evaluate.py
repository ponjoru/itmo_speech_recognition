import csv
import time
import jiwer
import datetime
import torchaudio
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable
from wav2vec2decoder import Wav2Vec2Decoder


def read_dataset(data_dir):
    manifest_path = str(Path(data_dir) /  "manifest.csv")
    samples = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            filename = Path(row["path"]).name
            audio_path = str(Path(data_dir) / filename)
            samples.append((audio_path, row["text"]))
    return samples
    

def eval_decoder(decoder, samples, method):
    results = {"refs": [], "hyps": []}
    
    t0 = time.time()
    for i, (audio_path, reference) in tqdm(enumerate(samples), total=len(samples)):
        audio_input, sr = torchaudio.load(audio_path)
        assert sr == 16000, f"Expected 16 kHz, got {sr} Hz"

        hyp = decoder.decode(audio_input, method)
        results["refs"].append(reference)
        results["hyps"].append(hyp)
    t1 = time.time()
    
    td = datetime.timedelta(seconds=t1 - t0)
    
    t = (datetime.datetime.min + td).time()
    t = t.strftime("%H:%M:%S")
    wer = jiwer.wer(results["refs"], results["hyps"])
    cer = jiwer.cer(results["refs"], results["hyps"])
    return wer, cer, t


# --------------------------------- Task functions -------------------------------------
def task1():
    decoder = Wav2Vec2Decoder(lm_model_path=None)
    
    samples = read_dataset("data/hw_2/librispeech_test_other")

    wer, cer, _ = eval_decoder(decoder, samples, method='greedy')
    
    t = PrettyTable(field_names=['method', 'WER', 'CER'])
    t.add_row(['greedy', f'{wer:.2%}', f'{cer:.2%}'])
    print(t)
    
def task2():
    method = 'beam'
    samples = read_dataset("data/hw_2/librispeech_test_other")
    results = []
    for beam_width in [1, 3, 10, 25, 50]:
        decoder = Wav2Vec2Decoder(beam_width=beam_width, lm_model_path=None)
        
        wer, cer, dt = eval_decoder(decoder, samples, method=method)
        
        results.append({'method': method, 'beam_width': beam_width, 'wer': f'{wer:.2%}', 'cer': f'{cer:.2%}', 'dt': dt})
    
    t = PrettyTable(field_names=list(results[0].keys()))
    for r in results:
        t.add_row(list(r.values()))
    print(t)

def task3():
    method = 'greedy'
    samples = read_dataset("data/hw_2/librispeech_test_other")
    results = []
    for temperature in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        decoder = Wav2Vec2Decoder(temperature=temperature, lm_model_path=None)
        
        wer, cer, _ = eval_decoder(decoder, samples, method='greedy')
        
        results.append({'method': method, 'temperature': temperature, 'WER': f'{wer:.2%}', 'CER': f'{cer:.2%}'})
    
    
    t = PrettyTable(field_names=list(results[0].keys()))
    for r in results:
        t.add_row(list(r.values()))
    print(t)
    
def task4():
    ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    BETAS = [0.0, 0.5, 1.0, 1.5]
    method = 'beam_lm'
    beam_width = 25
    lm_model_path = "weights/3-gram.pruned.1e-7.arpa.gz"
    samples = read_dataset("data/hw_2/librispeech_test_other")
    results = []
    
    total = len(ALPHAS) * len(BETAS)
    i = 1
    for alpha in ALPHAS:
        for beta in BETAS:
            print(f'Procesing sweep: {i}/{total}')
            decoder = Wav2Vec2Decoder(alpha=alpha, beta=beta, beam_width=beam_width, lm_model_path=lm_model_path)
            wer, cer, _ = eval_decoder(decoder, samples, method=method)
            results.append({'method': method, 'alpha': alpha, 'beta': beta, 'WER': f'{wer:.2%}', 'CER': f'{cer:.2%}'})
            i += 1
    
    t = PrettyTable(field_names=list(results[0].keys()))
    for r in results:
        t.add_row(list(r.values()))
    print(t)
    
def task5():
    alpha = 0.5
    beta = 1.5
    beam_width = 25
    
    method = 'beam_lm'
    samples = read_dataset("data/hw_2/librispeech_test_other")
    results = []
    
    decoder = Wav2Vec2Decoder(beam_width=beam_width, alpha=alpha, beta=beta, lm_model_path="weights/3-gram.pruned.1e-7.arpa.gz")
    wer, cer, _ = eval_decoder(decoder, samples, method=method)
    results.append({'method': method, 'alpha': alpha, 'beta': beta, 'WER': f'{wer:.2%}', 'CER': f'{cer:.2%}', 'LM': '3-gram'})

    decoder = Wav2Vec2Decoder(beam_width=beam_width, alpha=alpha, beta=beta, lm_model_path="weights/4-gram.arpa.gz")
    wer, cer, _ = eval_decoder(decoder, samples, method=method)
    results.append({'method': method, 'alpha': alpha, 'beta': beta, 'WER': f'{wer:.2%}', 'CER': f'{cer:.2%}', 'LM': '4-gram'})

    t = PrettyTable(field_names=list(results[0].keys()))
    for r in results:
        t.add_row(list(r.values()))
    print(t)

def task6():
    ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    BETAS = [0.0, 0.5, 1.0, 1.5]
    beam_width = 25
    method = 'beam_lm_rescore'
    lm_model_path = "weights/3-gram.pruned.1e-7.arpa.gz"
    samples = read_dataset("data/hw_2/librispeech_test_other")
    results = []
    
    total = len(ALPHAS) * len(BETAS)
    i = 1
    for alpha in ALPHAS:
        for beta in BETAS:
            print(f'Procesing sweep: {i}/{total}')
            decoder = Wav2Vec2Decoder(beam_width=beam_width, alpha=alpha, beta=beta, lm_model_path=lm_model_path)
            wer, cer, dt = eval_decoder(decoder, samples, method=method)
            results.append({'method': method, 'alpha': alpha, 'beta': beta, 'WER': f'{wer:.2%}', 'CER': f'{cer:.2%}', 'dt': dt})

            i += 1
    t = PrettyTable(field_names=list(results[0].keys()))
    for r in results:
        t.add_row(list(r.values()))
    print(t)

def task6b():
    beam_width = 25
    samples = read_dataset("data/hw_2/librispeech_test_other")
    results = []
    
    decoders = {
        'beam': Wav2Vec2Decoder(beam_width=10, lm_model_path=None),
        'beam_lm': Wav2Vec2Decoder(beam_width=beam_width, alpha=0.5, beta=1.5, lm_model_path="weights/3-gram.pruned.1e-7.arpa.gz"),
        'beam_lm_rescore': Wav2Vec2Decoder(beam_width=beam_width, alpha=0.1, beta=1.0, lm_model_path="weights/3-gram.pruned.1e-7.arpa.gz"),
    }
    
    results = []
    for i, (audio_path, reference) in tqdm(enumerate(samples), total=len(samples)):
        audio_input, sr = torchaudio.load(audio_path)
        assert sr == 16000, f"Expected 16 kHz, got {sr} Hz"

        preds = {}
        for method in decoders.keys():
            hyp = decoders[method].decode(audio_input, method)
            preds[method] = hyp
        preds['ref'] = reference
        
        results.append(preds)
    
    with open('results.txt', 'w') as fp:
        fp.write('='.center(50, '=') + '\n')
        for res in results:
            if res['beam'] != res["beam_lm"] or res['beam'] != res["beam_lm_rescore"]:
                fp.write(f'{"REF".rjust(5)}: {res["ref"]}\n')
                fp.write(f'{"BEAM".rjust(5)}: {res["beam"]}\n')
                fp.write(f'{"SF".rjust(5)}: {res["beam_lm"]}\n')
                fp.write(f'{"RS".rjust(5)}: {res["beam_lm_rescore"]}\n')
            endline = '='.center(50, '=')
            endline += '\n'
            fp.write(endline)

def task7():
    beam_width = 25
    datasets = {
        'LibriSpeech': read_dataset("data/hw_2/librispeech_test_other"),
        'Earnings 22': read_dataset("data/hw_2/earnings22_test"),
    }
    
    decoders = {
        'greedy': Wav2Vec2Decoder(lm_model_path=None),
        'beam': Wav2Vec2Decoder(beam_width=10, lm_model_path=None),
        'beam_lm': Wav2Vec2Decoder(beam_width=beam_width, alpha=0.5, beta=1.5, lm_model_path="weights/3-gram.pruned.1e-7.arpa.gz"),
        'beam_lm_rescore': Wav2Vec2Decoder(beam_width=beam_width, alpha=0.5, beta=1.0, lm_model_path="weights/3-gram.pruned.1e-7.arpa.gz"),
    }
    
    results = []
    for method in decoders.keys():
        res_dict = {'Method': method}
        for ds_name in datasets.keys():
            wer, cer, _ = eval_decoder(decoders[method], datasets[ds_name], method)
            res_dict[f'{ds_name} WER'] = f'{wer:.2%}'
            res_dict[f'{ds_name} CER'] = f'{cer:.2%}'
        
        results.append(res_dict)

    t = PrettyTable(field_names=list(results[0].keys()))
    for r in results:
        t.add_row(list(r.values()))
    print(t)
    

def task7b():
    beam_width = 25
    decoders = {
        'greedy': Wav2Vec2Decoder(lm_model_path=None),
        'beam_lm': Wav2Vec2Decoder(beam_width=beam_width, alpha=0.5, beta=1.5, lm_model_path="weights/3-gram.pruned.1e-7.arpa.gz"),
    }
    
    samples = read_dataset("data/hw_2/earnings22_test")
    results_greedy = []
    results_sf = []
    for temperature in {0.5, 1.0, 1.5, 2.0}:
        method = 'greedy'
        decoders[method].temperature = temperature
        wer, cer, _ = eval_decoder(decoders[method], samples, method=method)
        results_greedy.append({'method': method, 'temperature': temperature, 'WER': f'{wer:.2%}', 'CER': f'{cer:.2%}'})
        
        method = 'beam_lm'
        decoders[method].temperature = temperature
        wer, cer, _ = eval_decoder(decoders[method], samples, method=method)
        results_sf.append({'method': method, 'temperature': temperature, 'WER': f'{wer:.2%}', 'CER': f'{cer:.2%}'})
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    axes = axes.ravel()

    axes[0].plot([r['temperature'] for r in results_greedy], [r['WER'] for r in results_greedy])
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("WER")
    axes[0].set_title("Greedy WER vs T")
    
    axes[1].plot([r['temperature'] for r in results_sf], [r['WER'] for r in results_sf])
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("WER")
    axes[1].set_title("Shallow Fusion WER vs T")
    
    fig.savefig('results_task7b.jpg')
    
    t = PrettyTable(field_names=list(results_sf[0].keys()))
    for r in results_sf:
        t.add_row(list(r.values()))
    print(t)
    

def task9():
    beam_width = 25
    decoders_vanilla = {
        'beam_lm': Wav2Vec2Decoder(beam_width=beam_width, alpha=0.5, beta=1.5, lm_model_path="weights/3-gram.pruned.1e-7.arpa.gz"),
        'beam_lm_rescore': Wav2Vec2Decoder(beam_width=beam_width, alpha=0.5, beta=1.0, lm_model_path="weights/3-gram.pruned.1e-7.arpa.gz"),
    }
    decoders_financial = {
        'beam_lm': Wav2Vec2Decoder(beam_width=beam_width, alpha=0.5, beta=1.0, lm_model_path="weights/financial-3gram.arpa.gz"),
        'beam_lm_rescore': Wav2Vec2Decoder(beam_width=beam_width, alpha=0.5, beta=1.0, lm_model_path="weights/financial-3gram.arpa.gz"),
    }
    
    datasets = {
        'LibriSpeech': read_dataset("data/hw_2/librispeech_test_other"),
        'Earnings 22': read_dataset("data/hw_2/earnings22_test"),
    }
    

    results = []
    for method in decoders_vanilla.keys():
        res_dict = {'Method': method, 'Model': 'LibriSpeech 3-gram'}
        for ds_name in datasets.keys():
            wer, cer, _ = eval_decoder(decoders_vanilla[method], datasets[ds_name], method)
            res_dict[f'{ds_name} WER'] = f'{wer:.2%}'
            res_dict[f'{ds_name} CER'] = f'{cer:.2%}'
        
        results.append(res_dict)
        
    for method in decoders_financial.keys():
        res_dict = {'Method': method, 'Model': 'Financial 3-gram'}
        for ds_name in datasets.keys():
            wer, cer, _ = eval_decoder(decoders_financial[method], datasets[ds_name], method)
            res_dict[f'{ds_name} WER'] = f'{wer:.2%}'
            res_dict[f'{ds_name} CER'] = f'{cer:.2%}'
        
        results.append(res_dict)

    t = PrettyTable(field_names=list(results[0].keys()))
    for r in results:
        t.add_row(list(r.values()))
    print(t)
    

if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    task6b()
    task7()
    task7b()
    task9()
