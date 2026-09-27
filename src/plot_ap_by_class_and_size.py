"""
Genera fig_ap_clase_tamano.png a partir de los valores de AP de tus tablas
(validacion). No usa datos externos: solo los numeros que ya reportas.
Ejecutar:  python generar_fig_ap_clase_tamano.py
Guarda la figura en la carpeta actual; copiala luego a memoria_figs/.
"""
import numpy as np
import matplotlib.pyplot as plt
 
# AP@[0.5:0.95] por clase (de tu tabla de AP por clase, validacion)
clases = ['car', 'motorcycle', 'bicycle', 'truck', 'trafficcone', 'bus', 'human']
ap     = [0.618,  0.574,        0.531,     0.501,   0.483,         0.490, 0.438]
orden = np.argsort(ap)[::-1]
clases = [clases[i] for i in orden]
ap     = [ap[i] for i in orden]
 
# AP@[0.5:0.95] por tamano (de tu tabla de metricas COCO)
ap_size = [0.340, 0.597, 0.756]
 
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
 
b1 = a1.bar(clases, ap, color='#1f77b4')
a1.set_title('AP@[0.5:0.95] por clase')
a1.set_ylabel('AP@[0.5:0.95]')
a1.set_ylim(0, 0.7)
a1.axhline(0.519, ls='--', color='gray', lw=1)
a1.text(len(clases) - 0.5, 0.527, 'mAP = 0,519', color='gray', ha='right', fontsize=9)
for r, v in zip(b1, ap):
    a1.text(r.get_x() + r.get_width() / 2, v + 0.01, f'{v:.3f}'.replace('.', ','),
            ha='center', fontsize=8)
a1.tick_params(axis='x', rotation=35)
 
a2.bar(['pequeno', 'mediano', 'grande'], ap_size,
       color=['#d62728', '#ff7f0e', '#2ca02c'])
a2.set_title('AP@[0.5:0.95] por tamano de objeto')
a2.set_ylabel('AP@[0.5:0.95]')
a2.set_ylim(0, 0.8)
for i, v in enumerate(ap_size):
    a2.text(i, v + 0.01, f'{v:.3f}'.replace('.', ','), ha='center', fontsize=9)
 
plt.tight_layout()
plt.savefig('fig_ap_clase_tamano.png', dpi=150, bbox_inches='tight')
print('Generada fig_ap_clase_tamano.png')