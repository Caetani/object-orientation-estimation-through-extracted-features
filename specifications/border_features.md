#### Descritores de borda (brutos):
1. Perímetro externo [px]: $p_{ext}$
1. Perímetro interno total [px]: $p_{int}$
1. Perímetro total [px]: $p_{total}$
1. Número de buracos [buracos]: $n_h$
1. Área [$\text{px}^2$]: $A$
1. Posição x do centróide [px]: $C_x$
1. Posição y do centróide [px]: $C_y$
1. Altura da bounding box [px]: $h_{bbox}$
1. Largura da bounding box [px]: $w_{bbox}$
1. Posição x do centro da bounding box [px]: $x_c$
1. Posição y do centro da bounding box [px]: $y_c$
1. Comprimento do eixo principal [px]: $l_{major}$
1. Ângulo eixo principal [graus]: $\theta_{major}$
1. Comprimento do eixo menor [px]: $l_{minor}$

#### Características para o modelo:
1. *Compactness* do perímetro externo [$\frac{\text{px}^2}{\text{px}^2}$]: $\textit{compactness}_{ext} \frac{p_{ext}^2}{A}$
1. *Compactness* do perímetro interno total [$\frac{\text{px}^2}{\text{px}^2}$]: $\textit{compactness}_{int} = \frac{p_{int}^2}{A}$
1. *Compactness* do perímetro total [$\frac{\text{px}^2}{\text{px}^2}$]: $\textit{compactness}_{total} = \frac{p_{total}^2}{A}$
1. Distância do centroide [px]: $d_c = \sqrt{\Delta x^2 + \Delta y^2}$
1. Ângulo do centróide [graus]: $\theta_c = \textit{atan} \left( \frac{\Delta x}{\Delta y} \right)$
1. Seno do ângulo do centróide: $\textit{sin}(\theta_c)$
1. Cosseno do ângulo do centróide: $\textit{cos}(\theta_c)$
1. *Aspect Ratio* [px/px] = $\frac{h_{bbox}}{w_{bbox}}$
1. *Eccentricity* [px/px] =  $\sqrt{1 - \left( \frac{l_{minor}}{l_{major}} \right)^2}$
1. Seno do ângulo do eixo principal: $\textit{sin}(\theta_{major})$
1. Cosseno do ângulo do eixo principal: $\textit{cos}(\theta_{major})$
1. HoG - Frequência relativa de 0 graus a 30 graus
1. HoG - Frequência relativa de 30 graus a 60 graus
1. HoG - Frequência relativa de 60 graus a 90 graus
1. HoG - Frequência relativa de 90 graus a 120 graus
1. HoG - Frequência relativa de 120 graus a 150 graus
1. HoG - Frequência relativa de 150 graus a 180 graus
1. HoG - Frequência relativa de 180 graus a 210 graus
1. HoG - Frequência relativa de 210 graus a 240 graus
1. HoG - Frequência relativa de 240 graus a 270 graus
1. HoG - Frequência relativa de 270 graus a 300 graus
1. HoG - Frequência relativa de 300 graus a 330 graus
1. HoG - Frequência relativa de 330 graus a 360 graus
1. Momento invariante 1: $\phi_1$
1. Momento invariante 2: $\phi_2$
1. Momento invariante 3: $\phi_3$
1. Momento invariante 4: $\phi_4$
1. Momento invariante 5: $\phi_5$
1. Momento invariante 6: $\phi_6$
1. Momento invariante 7: $\phi_7$


Onde:
- $\Delta x = \frac{C_x - x_c}{w_{bbox}}$
- $\Delta y = \frac{C_y - y_c}{h_{bbox}}$