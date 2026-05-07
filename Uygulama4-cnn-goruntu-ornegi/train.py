import os
import csv
import json
import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from network import SimpleCNN


def set_seed(seed: int):
    """Tüm rastgelelik kaynaklarını sabitler — sonuçların tekrarlanabilir olmasını sağlar."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Belirleyici (deterministic) cuDNN kernelleri kullan
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
"""
Bu kod CIFAR veri kümesini kullanarak bir görüntü sınıflandırma
CNN'i eğitir.
"""

def train(model, device, loader, criterion, optimizer, epoch):
    # Modeli eğitim moduna geçirir
    model.train()
    running_loss = 0.0  # Bir döngü (epoch) boyunca biriken kayıp değeri
    correct = 0         # Eğitimde doğru sınıflandırma sayısı

    # Veri kümesini batch batch dolaş
    for batch_idx, (data, target) in enumerate(loader):
        # Veriyi ve hedefleri uygun cihaza (GPU/CPU) taşır
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # Önceki adımın gradyanlarını sıfırlar
        output = model(data)   # Modelden tahminleri alır
        loss = criterion(output, target)  # Kayıp fonksiyonunu hesaplar
        loss.backward()        # Geriye yayılım yaparak gradyanları hesaplar
        optimizer.step()       # Ağırlıkları günceller

        # Kayıp değerini Python float olarak toplar
        running_loss += loss.detach().item()

        # Eğitim doğruluğu için tahminleri topla
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

        # Her 100 batch'te bir ara çıktı verir
        if batch_idx % 100 == 0:
            print(f"Döngü {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}]  Kayıp: {loss.detach().item():.4f}")

    # Ortalama kayıp ve eğitim doğruluğunu hesapla
    avg_loss = running_loss / len(loader)
    train_accuracy = 100.0 * correct / len(loader.dataset)
    print(f"Döngü {epoch} Eğitim -> Kayıp: {avg_loss:.4f}, Doğruluk: %{train_accuracy:.2f}")
    return avg_loss, train_accuracy


def save_confusion_matrix(cm, class_names, out_path, title):
    """Karışıklık matrisini PNG olarak kaydeder."""
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8),
                                    max(5, len(class_names) * 0.7)))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='Gerçek',
           xlabel='Tahmin',
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Hücrelerin içine sayıları yaz
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate(model, device, loader, criterion, num_classes, epoch=None,
             split_name="Test", class_names=None, cm_dir="confusion_matrices"):
    """Test/Doğrulama veri kümesinde modeli değerlendirir ve metrikleri yazar."""
    # Modeli değerlendirme moduna geçirir (Dropout, BatchNorm etkisiz)
    model.eval()
    total_loss = 0.0  # Toplam kayıp
    correct = 0       # Doğru sınıflandırma sayısı

    # Sınıf bazlı sayaçlar:
    # tp: doğru pozitif, fp: yanlış pozitif, fn: yanlış negatif
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    support = [0] * num_classes  # Her sınıftaki gerçek örnek sayısı

    # Karışıklık matrisi: confusion[gerçek][tahmin]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Gradyan hesaplamayı kapatır (memory ve hız optimizasyonu)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)        # Model çıktısını al
            loss = criterion(output, target)  # Kayıp hesapla
            total_loss += loss.detach().item()  # Kayıp değerini topla

            # Tahmin edilen sınıfı bul ve doğru sayısını güncelle
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # Sınıf bazlı TP/FP/FN sayaçlarını ve karışıklık matrisini güncelle
            for t, p in zip(target.view(-1), pred.view(-1)):
                t_i, p_i = t.item(), p.item()
                support[t_i] += 1
                confusion[t_i, p_i] += 1
                if t_i == p_i:
                    tp[t_i] += 1
                else:
                    fp[p_i] += 1   # p_i sınıfı yanlış pozitif aldı
                    fn[t_i] += 1   # t_i sınıfı yanlış negatif aldı

    # Ortalama kayıp ve genel doğruluğu hesapla
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / len(loader.dataset)

    # Sınıf bazlı precision, recall, F1
    precisions = []
    recalls = []
    f1s = []
    for i in range(num_classes):
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # Macro (sınıf başına eşit ağırlıklı) ortalamalar
    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    macro_f1 = sum(f1s) / num_classes

    epoch_str = f"Döngü {epoch} " if epoch is not None else ""
    print(f"{epoch_str}{split_name} -> Kayıp: {avg_loss:.4f}, Doğruluk: %{accuracy:.2f} ({correct}/{len(loader.dataset)})")
    print(f"  Macro -> Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")

    # Sınıf bazlı rapor
    print(f"  Sınıf bazlı metrikler:")
    print(f"    {'Sınıf':<8}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Destek':>10}")
    for i in range(num_classes):
        print(f"    {i:<8}{precisions[i]:>12.4f}{recalls[i]:>10.4f}{f1s[i]:>10.4f}{support[i]:>10}")

    # Confusion matrix'i kaydet
    os.makedirs(cm_dir, exist_ok=True)
    labels = class_names if class_names is not None else [str(i) for i in range(num_classes)]
    suffix = f"epoch_{epoch}" if epoch is not None else split_name.lower()
    cm_path = os.path.join(cm_dir, f"confusion_matrix_{suffix}.png")
    title = f"{split_name} Confusion Matrix" + (f" (Döngü {epoch})" if epoch is not None else "")
    save_confusion_matrix(confusion, labels, cm_path, title)
    print(f"  Confusion matrix kaydedildi: {cm_path}\n")

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'precisions': precisions,
        'recalls': recalls,
        'f1s': f1s,
        'support': support,
    }

def main():
    # ---------- Yapılandırma ----------
    data_dir = 'data_cifar'  # Eğitim ve doğrulama klasörlerini içeren kök dizin
    batch_size = 32          # Batch (yığın) boyutu
    epochs = 10              # Toplam epoch (döngü) sayısı
    learning_rate = 0.001    # Öğrenme oranı
    seed = 42                # Tekrarlanabilirlik için rastgelelik tohumu
    run_tag = ''             # İsteğe bağlı: run_dir adına eklenecek özel etiket

    # ---------- Run klasörü ----------
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parts = [timestamp, f"bs{batch_size}", f"lr{learning_rate}",
             f"ep{epochs}", f"seed{seed}"]
    if run_tag:
        parts.append(run_tag)
    run_dir = os.path.join('runs', '_'.join(parts))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run klasörü: {run_dir}")

    # ---------- Seed sabitleme ----------
    set_seed(seed)
    # DataLoader shuffle'ı için ayrı bir generator
    g = torch.Generator()
    g.manual_seed(seed)
    print(f"Seed sabitlendi: {seed}")

    # ---------- Cihaz Ayarı (GPU/MPS/CPU) ----------
    device = torch.device('cuda' if torch.cuda.is_available() \
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}\n")

    # ---------- Veri Dönüşümleri ----------
    transform = transforms.Compose([
        # Görüntüleri tensöre çevir
        transforms.ToTensor(),
        # ImageNet istatistikleri ile normalizasyon
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---------- Dataset ve DataLoader Oluşturma ----------
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

     # ---------- Model, Kayıp Fonksiyonu ve Optimizatör ----------
    num_classes = len(train_dataset.classes)
    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Sınıf sayısı: {num_classes}\nSınıflar: {train_dataset.classes}\n")

    # ---------- Hiperparametreleri config.json olarak kaydet ----------
    config = {
        'timestamp': timestamp,
        'data_dir': data_dir,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'seed': seed,
        'optimizer': 'Adam',
        'loss': 'CrossEntropyLoss',
        'device': str(device),
        'num_classes': num_classes,
        'classes': train_dataset.classes,
        'model': {
            'name': type(model).__name__,
            # network.py içindeki katmanları string olarak kaydet — kanal sayısı vb.
            # değişiklikler burada görünür ve sonradan denemeleri ayırt etmeyi sağlar.
            'architecture': str(model),
        },
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Metrik kayıt dosyaları: epoch özeti ve sınıf bazlı detay
    metrics_path = os.path.join(run_dir, 'metrics.csv')
    per_class_path = os.path.join(run_dir, 'metrics_per_class.csv')
    cm_dir = os.path.join(run_dir, 'confusion_matrices')

    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_loss', 'train_accuracy',
            'test_loss', 'test_accuracy',
            'test_macro_precision', 'test_macro_recall', 'test_macro_f1',
        ])

    with open(per_class_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'class_index', 'class_name',
            'precision', 'recall', 'f1', 'support',
        ])

    # Training and validation loop
    for epoch in range(1, epochs + 1):
        print(f"===== Döngü {epoch}/{epochs} =====")
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
        # Her döngü sonunda test/doğrulama verisinde metrikleri ölç
        results = evaluate(
            model, device, val_loader, criterion, num_classes,
            epoch=epoch, split_name="Test",
            class_names=train_dataset.classes,
            cm_dir=cm_dir,
        )

        # Epoch özetini ekle
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{results['loss']:.6f}", f"{results['accuracy']:.4f}",
                f"{results['macro_precision']:.6f}",
                f"{results['macro_recall']:.6f}",
                f"{results['macro_f1']:.6f}",
            ])

        # Sınıf bazlı satırları ekle
        with open(per_class_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(num_classes):
                writer.writerow([
                    epoch, i, train_dataset.classes[i],
                    f"{results['precisions'][i]:.6f}",
                    f"{results['recalls'][i]:.6f}",
                    f"{results['f1s'][i]:.6f}",
                    results['support'][i],
                ])
    print(f"Metrikler kaydedildi: {metrics_path}, {per_class_path}")

    # Save the trained model
    model_path = os.path.join(run_dir, 'simple_cnn.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Eğitim tamamlandı. Tüm çıktılar: {run_dir}")


if __name__ == '__main__':
    main()
