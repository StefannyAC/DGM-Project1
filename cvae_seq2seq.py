# cvae_seq2seq.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Implementación de un CVAE condicional para piano-rolls con eventos asociados.
# El piano-roll es una representación binaria 128xT (notas vs tiempo).
# El CVAE está condicionado en una etiqueta de clase (género musical).
# ============================================================
# CVAE condicional por género musical compatible con data_pipeline.py
# usando arquitectura seq2seq con RNNs
# - X_pr: (B, 128, T), corresponde a piano-rolls binarios
# - events: (B, N, 3) con padding; el encoder ignora filas cero , corresponde al event-based encoding
# - cond: (B, 1) o (B,) Long (id de género en {0..3}), corresponde a la condición
# ============================================================

import math # solo para clamp de logvar
import torch # Para tensores
import torch.nn as nn # Para definir modelos
import torch.nn.functional as F # Para funciones de activación y pérdidas

# ------------ Event Encoder ------------
class EventEncoder(nn.Module):
    """
    Función para codificar una secuencia (timestamp, pitch, velocity) de eventos de dimensión 'in_dim'
    en un vector fijo de tamaño 'ev_embed', ignorando filas de padding compuestas solo por ceros.

    La codificación se realiza aplicando un MLP por evento (proyección por fila) y luego
    promediando únicamente sobre las filas válidas (no-cero). Si una secuencia no tiene
    eventos válidos, se devuelve el vector cero.

    Args:
        in_dim: Número de características por evento (por defecto 3).
        ev_embed: Dimensión del embedding de salida que representa a toda la secuencia.

    Returns:
        torch.Tensor: Tensor de forma '(B, ev_embed)' con una representación por secuencia.
    """
    def __init__(self, in_dim=3, ev_embed=64):
        """Función para inicializar las capas (MLP) y registrar la dimensión de salida.
        """
        super().__init__() # Inicializa correctamente nn.Module (registra submódulos/buffers).
        # MLP por-evento: cada fila (evento) se procesa de forma independiente.
        self.mlp = nn.Sequential( 
            nn.Linear(in_dim, 16), nn.ReLU(inplace=True), # Proyección lineal: R^{in_dim} -> R^{16}. Luego, ReLU, que corresponde a una activación no lineal, usamos inplace para ahorrar algo de memoria.
            nn.Linear(16, ev_embed), nn.ReLU(inplace=True), # Proyección lineal: R^{16} -> R^{ev_embed}. Luego, ReLU, que mantiene activaciones no negativas.
        )
        self.ev_embed = ev_embed # Guardamos para crear tensores cero cuando sea necesario.

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Función forward que procesa una secuencia de eventos y devuelve su representación fija.
        Además, ignora filas de padding compuestas solo por ceros.

        Args:
            events: (B, N, 3) con posibles filas de ceros (padding). También acepta (N,3) y lo expande a (1,N,3).
            -  B: tamaño de batch
            -  N: número máximo de eventos en la secuencia
            -  3: características por evento (timestamp, pitch, velocity)
        Returns:
            torch.Tensor: Tensor de forma '(B, ev_embed)' con la representación por secuencia
        """
        # Si llega sin dimensión de batch, expandimos a (1, N, in_dim) para unificar el flujo.
        if events.ndim == 2:  # (N,3) -> (1,N,3)
            events = events.unsqueeze(0)

        # Si el tensor no contiene elementos (caso extremo), devolvemos embedding cero.    
        if events.numel() == 0:
            # sin eventos: regresa vector cero
            B = 1 if events.ndim == 2 else events.size(0)
            return torch.zeros(B, self.ev_embed, device=events.device, dtype=events.dtype)

        # Aseguramos tipo flotante: nn.Linear requiere tensores flotantes
        if not torch.is_floating_point(events):
            events = events.float()

        # Desempaquetamos dimensiones
        B, N, _ = events.shape

        # Caso sin eventos válidos: devolvemos vector cero
        if N == 0:
            return torch.zeros(B, self.ev_embed, device=events.device, dtype=events.dtype)

        # máscara: 1.0 si la fila tiene algún valor != 0 (evento válido), 0.0 si es fila de padding
        mask = (events.abs().sum(dim=-1) > 0).float()         # (B, N)
        # Extraemos características por evento con el MLP (aplicación por fila).
        feats = self.mlp(events)                               # (B, N, ev_embed)
        # Anulamos contribución de padding multiplicando por la máscara (broadcast en la última dim) y sumamos en N.
        summed = (feats * mask.unsqueeze(-1)).sum(dim=1)      # (B, ev_embed)
        # Contamos eventos válidos por secuencia para hacer el promedio. Evitamos división por cero con clamp_min.
        counts = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1) # evitar /0

        # Promediamos sobre eventos válidos, devolviendo vector cero si no hay eventos válidos.
        return summed / counts                                 # (B, ev_embed)

# ----------------- CVAE: Seq2Seq -----------------
class CVAE(nn.Module):
    """
    Función para modelar una CVAE (Conditional VAE) que reconstruye secuencias piano-roll
    (o embeddings de dimensión 128 por tiempo) condicionadas por etiquetas discretas y por un
    resumen de eventos E (event-based).
    Arquitectura seq2seq con RNNs:
        forward(X, E, y) -> (X_rec, mu, logvar)
        - encoder(X, E, y) -> (mu, logvar)
        - reparameterize(mu, logvar) -> z
        - decode(z, y) -> X_rec  (B,128,T)
        Donde:
            - X: entrada (B, 128, T), señales en [0,1] (piano-roll binario o continuo).
            - E: eventos (B, N, 3) para codificar estructura adicional (vía EventEncoder).
            - y: condición categórica (B,) o (B,1) en {0..cond_dim-1}, por defecto cond_dim=4.
            - z: variable latente (B, z_dim), mu/logvar de la distribución latente.
            - X_rec: reconstrucción de X (B, 128, T).
    """
    def __init__(
        self,
        z_dim=128,
        cond_dim=4,
        seq_len=32,
        ev_embed=64,
        cond_embed=16,
        enc_hid=256, 
        dec_hid=256
    ):
        """
        Función para inicializar las capas del CVAE.

        Args:
            z_dim: Dimensión de la variable latente z.
            cond_dim: Número de clases para la condición categórica y.
            seq_len: Longitud de la secuencia de piano-roll (T).
            ev_embed: Dimensión del embedding generado por el EventEncoder.
            cond_embed: Dimensión del embedding para la condición categórica y.
            enc_hid: Tamaño del estado oculto del GRU del encoder.
            dec_hid: Tamaño del estado oculto del GRU del decoder.
        """
        super().__init__() # Inicializa correctamente nn.Module (registra submódulos/buffers).
        self.z_dim = z_dim # dimensión latente
        self.seq_len = seq_len # longitud de secuencia T

        # Codificador de eventos (B, N, 3) -> (B, ev_embed). Ignora padding (filas cero)
        self.ev_encoder  = EventEncoder(in_dim=3, ev_embed=ev_embed)

        # Embedding para la condición y -> (B, 4)
        self.cond_embed  = nn.Embedding(cond_dim, cond_embed)

        # ---------------- Encoder RNN ----------------
        # GRU bidireccional que lee las features temporales (T pasos) y produce última h por dirección.
        # input_size=128 porque X vendrá como (B, T, 128) tras transponer.
        self.enc_rnn = nn.GRU(input_size=128, hidden_size=enc_hid, num_layers=1, bidirectional=True, batch_first=True)

        # Proyecciones para media y logvar de la distribución latente
        # [h_last_bidir (2*enc_hid), e_emb (ev_embed), y_emb (cond_embed)]
        enc_in = 2*enc_hid + ev_embed + cond_embed # De esta manera porque es bidireccional y concatena
        self.fc_mu     = nn.Linear(enc_in, z_dim) # Proyección a media de z
        self.fc_logvar = nn.Linear(enc_in, z_dim) # Proyección a logvar de z

        # ---------------- Decoder RNN ----------------
        # Proyección de [z, y_emb] para inicializar h0 del decodificador.
        self.h0_proj = nn.Linear(z_dim + cond_embed, dec_hid)              # estado inicial

        # GRU decodificador que recibe en cada paso [x_t (128), y_emb (cond_embed)].
        self.dec_rnn = nn.GRU(input_size=128 + cond_embed, hidden_size=dec_hid,
                              num_layers=1, batch_first=True) # 128 + cond_embed porque concatena 
        
        # Capa de salida por paso temporal: dec_hid -> 128 + Sigmoid para valores en [0,1].
        self.out = nn.Sequential(nn.Linear(dec_hid, 128), nn.Sigmoid())    # (B,T,128)

    # --- Encoder ---
    def encoder(self, X, E, y):
        """
        Función para codificar (X, E, y) en los parámetros de la distribución latente (mu, logvar).

        Args:
            X: Tensor de entrada (B, 128, T) con características por tiempo.
            E: Eventos (B, N, 3) con padding posible (filas cero). Se codifican a (B, ev_embed).
            y: Condición categórica (B,) o (B,1) con enteros en [0..cond_dim-1].

        Returns:
            tuple:
                - mu     : (B, z_dim) media de la latente.
                - logvar : (B, z_dim) log-varianza de la latente.
        """
        # X: (B,128,T) -> (B,T,128) para alimentar al GRU (batch_first=True)
        Xt = X.transpose(1, 2)

        # Pasar secuencia por la GRU bidireccional del encoder
        h_seq, h_last = self.enc_rnn(Xt)                 # h_last: (num_layers*2, B, enc_hid) = (2,B,enc_hid)
        # Concatenar las dos direcciones del último estado oculto
        h_last = torch.cat([h_last[-2], h_last[-1]], dim=1)  # (B,2*enc_hid)

        # Aseguramos y como vector (B,), por si viene como (B,1).
        if y.ndim == 2 and y.size(-1) == 1: y = y.squeeze(-1)

        # Embedding de la condición: (B,) -> (B, cond_embed).
        y_emb = self.cond_embed(y.long())                # (B,cond_embed)
        
        # Embedding de eventos: (B,N,3) -> (B, ev_embed)
        e_emb = self.ev_encoder(E)                       # (B,ev_embed)

        # Concatenar h_last, e_emb, y_emb y proyectar a mu/logvar
        h = torch.cat([h_last, e_emb, y_emb], dim=1)     # (B, enc_out_dim)

        # Proyecciones a mu y logvar de z
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)

        # Clamp de logvar para estabilidad numérica: evita varianzas extremas y explosiones en exp(*)
        logvar = logvar.clamp(min=-10.0, max=10.0)  
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Función para muestrear z mediante el truco de reparametrización: z = mu + sigma * eps.

        Args:
            mu: Media (B, z_dim).
            logvar: Log-varianza (B, z_dim).

        Returns:
            torch.Tensor: Muestra latente z (B, z_dim) estable numéricamente.
        """
        # std = exp(0.5 * logvar); se clampa para evitar underflow/overflow.
        std = torch.exp(0.5 * logvar).clamp(min=1e-6, max=50.0)
        # Ruido estándar ~ N(0, I) con el mismo shape que std.
        eps = torch.randn_like(std)
        # Reparametrización z ~ N(mu, std^2)
        z = mu + eps*std
        # Evitar NaN en z retornando 0.0 donde haya NaN, y clamp en +-50 para evitar inf.
        return torch.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0) # evitamos NaN

    # --- Decoder ---
    def decode(self, z, y, T=None, teacher=None):
        """
        Función para decodificar una latente z (condicionada por y) a una secuencia X_hat de longitud T.

        Soporta teacher forcing si se pasa 'teacher' (usa la secuencia real como entrada del RNN);
        si no, decodifica de forma autoregresiva.

        Args:
            z: Latente (B, z_dim).
            y: Condición categórica (B,) o (B,1).
            T: Longitud a generar; si None, usa 'self.seq_len'.
            teacher: Secuencia real para teacher forcing (B, 128, T) o None.

        Returns:
            torch.Tensor: Reconstrucción (B, 128, T) en [0,1].
        """
        B = z.size(0); T = T or self.seq_len # tamaño de batch y longitud T por defecto si no se especifica

        # Aseguramos y como vector (B,), por si viene como (B,1).
        if y.ndim == 2 and y.size(-1) == 1: 
            y = y.squeeze(-1)

        # Embedding de la condición: (B,) -> (B, cond_embed).
        y_emb = self.cond_embed(y.long())                    # (B,cond_embed)

        # Inicializar h0 del decodificador a partir de z y y_emb: h0 = tanh(W [z, y_emb])
        h0 = torch.tanh(self.h0_proj(torch.cat([z, y_emb], dim=1))).unsqueeze(0)  # (1,B,dec_hid)

        if teacher is not None:
            # ----- Teacher forcing -----
            # teacher: (B,128,T) -> (B,T,128+cond)
            dec_inp = torch.cat([teacher.transpose(1,2),
                                 y_emb.unsqueeze(1).repeat(1, T, 1)], dim=2) # Repetimos y_emb a lo largo del tiempo y lo concatenamos con la entrada por paso 
            # Ejecutamos la GRU con h0 fijo
            h_seq, _ = self.dec_rnn(dec_inp, h0)            # (B,T,dec_hid)

            # Proyección a dimensión 128 y Sigmoid a [0,1].
            Y = self.out(h_seq)                              # (B,T,128)
        else:
            # ----- Modo autoregresivo -----
            # y_t es la "semilla" inicial (todo ceros) que se irá actualizando con la salida previa.
            y_t = torch.zeros(B, 1, 128, device=z.device)
            cond_rep = y_emb.unsqueeze(1)                   # (B,1,cond_embed)
            h = h0; outs = []                               # estado oculto inicial y lista para salidas
            for _ in range(T):
                # Entrada en el paso t: concat([salida anterior], y_emb) a lo largo de la última dim.
                x_t = torch.cat([y_t, cond_rep], dim=2)     # (B,1,128+cond)
                # Un paso de GRU y actualización de estado oculto
                h_seq, h = self.dec_rnn(x_t, h) # h_seq: (B, 1, dec_hid), h: (1, B, dec_hid)
                # Proyección a dimensión 128 y Sigmoid a [0,1].
                y_t = self.out(h_seq)                       # (B,1,128) in [0,1]
                # Guardamos la salida del paso actual.
                outs.append(y_t)
            # Concatenamos las T salidas a lo largo del tiempo.    
            Y = torch.cat(outs, dim=1)                      # (B,T,128)
        # Volvemos a la convención (B, 128, T) para mantener coherencia con el resto del pipeline.
        return Y.transpose(1, 2)                             # (B,128,T)
    
    def forward(self, X, E, y, teacher_prob: float = 1.0):
        """
        Función para ejecutar el paso completo de la CVAE: codificar, muestrear z y decodificar.

        Args:
            X: Entrada objetivo (B, 128, T) usada para el cálculo de pérdida y/o teacher forcing.
            E: Eventos (B, N, 3) para el EventEncoder.
            y: Condición categórica (B,) o (B,1).
            teacher_prob: Probabilidad de usar teacher forcing durante entrenamiento
                                  (aquí se usa como interruptor simple > 0 -> usar X completo).

        Returns:
            tuple:
                - X_rec (B, 128, T): Reconstrucción.
                - mu    (B, z_dim) : Media de la latente.
                - logvar(B, z_dim) : Log-varianza de la latente.
        """
        # Parametrizamos la distribución q(z|X,E,y).
        mu, logvar = self.encoder(X, E, y)
        # Muestreamos z con el truco de reparametrización.
        z = self.reparameterize(mu, logvar)
        # Teacher forcing si estamos en training y se habilita (usa X como entrada del decodificador).
        teacher = X if self.training and teacher_prob > 0 else None
        # Decodificamos a longitud T = X.size(-1) para comparar 1:1 con el objetivo.
        X_rec = self.decode(z, y, T=X.size(-1), teacher=teacher)
        return X_rec, mu, logvar