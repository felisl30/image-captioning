import matplotlib.pyplot as plt


def show_sample(sample, title="MIMIC-CXR sample"):
    """
    Muestra una imagen del dataset junto con findings e impression.
    """

    image = sample["image"]
    findings = sample.get("findings")
    impression = sample.get("impression")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    ax.set_title(title)
    plt.show()

    print("FINDINGS:")
    print(findings)

    print("\nIMPRESSION:")
    print(impression)


def plot_text_lengths(texts, title="Distribución de longitudes"):
    """
    Grafica distribución de longitudes de textos.
    """

    lengths = [
        len(str(t))
        for t in texts
        if t is not None and str(t).strip() != ""
    ]

    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=50)
    plt.title(title)
    plt.xlabel("Cantidad de caracteres")
    plt.ylabel("Frecuencia")
    plt.show()

    print("Cantidad de textos válidos:", len(lengths))
    print("Longitud mínima:", min(lengths))
    print("Longitud máxima:", max(lengths))
    print("Longitud promedio:", sum(lengths) / len(lengths))
