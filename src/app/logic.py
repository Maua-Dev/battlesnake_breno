# Bem-vindo ao
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \   __\   __\  | _/ __ \ /  ___//    \__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\_____>
#
# =============================================================================
#  v2.1.0 — ALIMENTAÇÃO ESTRATÉGICA NA BUSCA 1v1 (sobre a v2.0.0)
# =============================================================================
#  Diagnóstico: na busca, a comida valia no máximo ~3.5 pontos (7/(1+t)) contra 3-10
#  pontos de território por jogada, não dependia de vida/fase/tamanho e quase não criava
#  gradiente de aproximação. PHASE_MODS e a urgência de vida só afetavam o desempate.
#  1. Valor de comida DINÂMICO dentro de Search.evaluate (interesse = base + (1-base)*urgência;
#     base depende da abertura, do tamanho relativo e da escassez de comida).
#  2. Qualidade por comida calculada UMA vez por jogada (hazard, beco, espaço depois de
#     comer, custo em vida, risco de head-to-head) e reutilizada em todas as folhas.
#  3. "Food commitment": a melhor comida segura e alcançável antes do rival ganha reforço,
#     e a raiz dá um bônus por progresso real (potencial: aproximar +, afastar -, ciclo = 0).
#  4. Recompensa de comer >= valor de ficar colado na comida (a cobra não "orbita" a comida).
#  5. PHASE_MODS agora governa comida (s_food), progresso (s_commit), território e pressão.
#  6. Anti-orbitação: memória curta das últimas cabeças; se repetir posições, sobe o interesse
#     em comida e penaliza revisitar casas (pequeno e limitado).
#  7. CORREÇÃO: empate por morte mútua valia 0.0 (= posição equilibrada) e a busca o escolhia
#     (duas cobras iguais entrando juntas na mesma comida). Agora DRAW é bem negativo.
#

# Documentação: https://docs.battlesnake.com


import heapq
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache

from .models import GameState, MoveResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Pt = tuple[int, int]

# --------------------------------------------------------------------------- #
# CONFIGURAÇÃO (tudo que você vai querer ajustar fica aqui)
# --------------------------------------------------------------------------- #

MOVES: dict[str, Pt] = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
MOVE_ORDER = ["up", "down", "left", "right"]
INF = 10**6
MAX_HEALTH = 100

# Liga/desliga a busca 1v1 (False = comportamento da v1.1.0: só pontuação).
USE_SEARCH = True

# Pesos da pontuação. Escala aproximada: "mortal" ~ 700-900, "importante" ~ 100-300,
# "desempate" ~ 10-40. Não são valores matematicamente ótimos: ajuste com partidas de teste.
WEIGHTS: dict[str, float] = {
    # --- sobrevivência ---
    "h2h_loss": 900.0,    # casa que uma cobra MAIOR pode alcançar: perderíamos o head-to-head
    "h2h_tie": 700.0,     # idem com cobra de MESMO tamanho: as duas morrem
    "dead_end": 700.0,    # penalidade máxima quando a região acessível é menor que o corpo
    "space": 200.0,       # espaço acessível (flood fill), saturando em um "conforto"
    "mobility": 20.0,     # por saída livre logo após o movimento (evita corredores de 1 casa)
    "tail_reach": 40.0,   # bônus por conseguir alcançar a própria cauda (rota de fuga)
    # --- território / posição ---
    "territory": 250.0,   # fatia do tabuleiro que chegamos ANTES dos adversários (Voronoi)
    "center": 25.0,       # leve preferência pelo centro (some quando precisa de comida)
    "edge": 12.0,         # leve penalidade por borda (menos opções de fuga)
    # --- comida / saúde ---
    "food": 300.0,        # valor da melhor comida alcançável (modulado por urgência)
    "starvation": 300.0,  # não chegaremos a nenhuma comida antes de morrer de fome
    "low_health": 60.0,   # penalidade crescente conforme a vida cai de LOW_HEALTH
    # --- hazards ---
    "hazard": 60.0,       # custo de entrar em hazard (cresce quando a vida está baixa)
    # --- adversários ---
    "kill": 250.0,        # chance de eliminar cobra MENOR via head-to-head
    "threat": 60.0,       # proximidade da cabeça de cobra maior/igual
    "hunt": 80.0,         # aproximação de cabeça de cobra menor (quando saudável)
    # --- tática de cauda ---
    "tail_follow": 120.0, # seguir a própria cauda quando o espaço está apertado
    # --- busca 1v1 (escalados por PHASE_MODS) ---
    "s_food": 50.0,       # pontos de uma comida segura, colada em mim, com interesse total
    "s_commit": 6.0,      # pontos por passo de progresso rumo à comida comprometida (raiz)
}

# Limiares e fatores (não são "pesos", mas também são ajustáveis).
TUNING: dict[str, float] = {
    "health_critical": 25,      # abaixo disso, urgência de comida = 1.0
    "health_comfort": 65,       # acima disso, urgência = 0.0
    "hazard_urgency_shift": 15, # em mapas com hazard a vida se esgota mais rápido
    "food_base": 0.25,          # interesse mínimo em comida (crescer) mesmo com vida cheia
    "food_base_ahead": 0.10,    # idem quando já somos bem maiores que todos
    "scarce_food": 2,           # com <= N comidas no mapa, comida vale um pouco mais
    "contested": 0.2,           # fator se um adversário chega antes (ou empata e é >= a nós)
    "hazard_food": 0.4,         # fator para comida dentro de hazard (se não urgente)
    "deadend_food": 0.4,        # fator para comida em beco (<= 1 saída)
    "low_health": 20,           # abaixo disso começamos a penalizar vida baixa
    "threat_range": 3,          # distância em que cabeças maiores nos assustam
    "hunt_range": 4,            # distância em que caçamos cabeças menores
    "space_comfort_min": 20,    # espaço "confortável" mínimo (cresce com o tamanho)
    "tail_min_len": 8,          # só seguimos a cauda ativamente com corpo desse tamanho
    "opening_turns": 12,        # turnos considerados "abertura"
    "time_budget": 0.4,         # fração do timeout que podemos gastar calculando (pontuação)
    "endgame_fill": 0.22,       # 1v1: corpos ocupando >= essa fração do tabuleiro = reta final
}

# Estratégia de comida na busca 1v1 (tudo que decide "quando vale crescer").
FOOD: dict[str, float] = {
    "base_open": 0.55,      # interesse mínimo em comida no turno 0 (decai até base_mid)
    "base_mid": 0.30,       # interesse mínimo com vida cheia, tamanho parecido
    "base_ahead": 0.12,     # idem quando já estou bem maior
    "ahead_margin": 3,      # segmentos a mais para contar como "bem maior"
    "behind_bonus": 0.12,   # +interesse se sou igual ou menor (crescer vence choques)
    "scarce_bonus": 0.10,   # +interesse com poucas comidas no mapa
    "early_turns": 30,      # turnos para o interesse de abertura decair até base_mid
    "h_min": 9.0,           # alcance (passos) do valor de comida com vida folgada
    "h_urgent": 20.0,       # alcance com vida crítica (comida distante passa a importar)
    "deny": 0.55,           # fração do valor descontada quando o RIVAL leva a comida
    "eat_extra": 0.10,      # comer vale (qualidade + extra) x pull: nunca menos que ficar colado
    "commit_boost": 1.5,    # reforço da comida comprometida
    "commit_min": 0.10,     # interesse x qualidade mínimos para comprometer
    "space_checks": 4,      # quantas comidas (as mais próximas) checam o espaço depois de comer
    "stall_min": 3,         # posições repetidas nas últimas jogadas = orbitando
    "stall_bonus": 0.20,    # +interesse em comida quando orbitando
    "revisit_pen": 1.5,     # pontos por revisitar uma casa recente (só orbitando; máx. 3x)
}

# Busca 1v1. Pesos da avaliação da busca: 1 ponto = 1 casa de território de vantagem.
SEARCH: dict[str, float] = {
    "time": 0.30,           # fração do timeout usada pela busca
    "cap_ms": 220.0,        # teto de tempo da busca
    "min_ms": 25.0,         # piso de tempo (mesmo com latência alta)
    "w_length": 3.0,        # por segmento a mais que a rival (ganha choques e território)
    "w_starve": 400.0,      # morro de fome antes de alcançar qualquer comida
    "w_starve_margin": 6.0, # por turno de folga abaixo de 8 até a comida mais próxima
    "w_cramped": 6.0,       # por casa que falta para o território igualar meu tamanho
    "w_hazard": 20.0,       # cabeça dentro de hazard
}
WIN = 100000
# Morte mútua (mesmo tamanho / dois na mesma casa). Antes valia 0.0, igual a uma posição equilibrada,
# e a busca chegava a ESCOLHER o empate (ex.: duas cobras entrando juntas na mesma comida).
# Agora é pior que qualquer posição viva (as notas heurísticas ficam bem abaixo de 10000 em
# módulo) e melhor que uma derrota forçada: só aceitamos o empate se o resto for perder.
DRAW = -(WIN // 10)
ROOT_EPS = 1e-6

# Multiplicadores por fase da partida (só altera o que for listado).
PHASE_MODS: dict[str, dict[str, float]] = {
    "opening": {"food": 1.3, "hunt": 0.3, "kill": 0.7, "s_food": 1.35, "s_commit": 1.2},
    "midgame": {},
    # reta final do 1v1 (à frente em tamanho ou tabuleiro cheio): mais território e pressão
    "endgame": {"hunt": 2.0, "territory": 1.3, "food": 0.8, "s_food": 0.9, "s_commit": 0.8},
}


# --------------------------------------------------------------------------- #
# INFO / START / END
# --------------------------------------------------------------------------- #

def info() -> dict:
    return {
        "apiversion": "1",
        "author": "",
        "color": "#8B0000",
        "head": "tiger-king",
        "tail": "hook",
        "version": "2.1.0",
    }


def start(state: GameState) -> None:
    logger.info("JOGO COMEÇOU (partida %s)", state.game.id)


def end(state: GameState) -> None:
    logger.info("FIM DE JOGO após %d turnos", state.turn)


# --------------------------------------------------------------------------- #
# MODELOS INTERNOS
# --------------------------------------------------------------------------- #

@dataclass
class Enemy:
    id: str
    head: Pt
    body: list
    length: int
    health: int
    next_cells: list = field(default_factory=list)  # casas que a cabeça pode ocupar no próximo turno


@dataclass
class Context:
    width: int
    height: int
    turn: int
    wrapped: bool
    constrictor: bool
    hazard_damage: int
    map_name: str
    my_head: Pt
    my_tail: Pt
    my_body: list
    my_len: int
    my_health: int
    enemies: list
    food: set
    hazards: set
    free_at: dict          # casa -> turno (após o movimento) em que deixa de estar ocupada
    deadline: float
    phase: str = "midgame"
    urgency: float = 0.0
    w: dict = field(default_factory=dict)              # pesos já ajustados pela fase
    enemy_arrival: dict = field(default_factory=dict)  # casa -> (turno de chegada, tamanho)
    max_enemy_len: int = 0
    timeout_ms: int = 500
    latency: float = 0.0   # ms medidos pelo jogo na jogada anterior (rede + nosso cálculo)
    recent: list = field(default_factory=list)   # últimas cabeças (a atual é a última)
    stall: int = 0         # quantas posições se repetiram nessa janela (orbitando)


@dataclass
class FloodResult:
    arrival: dict          # casa -> turno em que chegamos nela
    count: int
    tail_reachable: bool


@dataclass
class Routes:
    cost: dict             # casa -> custo em PONTOS DE VIDA (1 por passo + dano de hazard)
    turn: dict             # casa -> turno de chegada pelo caminho escolhido


@dataclass
class MoveEvaluation:
    move: str
    pos: Pt
    score: float
    parts: dict


# --------------------------------------------------------------------------- #
# LEITURA SEGURA DO STATE E CONSTRUÇÃO DO CONTEXTO
# --------------------------------------------------------------------------- #

def _dig(obj, *names, default=None):
    """Acessa obj.a.b.c (ou dict['a']['b']) sem estourar erro; devolve default se faltar."""
    for name in names:
        if obj is None:
            return default
        obj = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return default if obj is None else obj


def _pt(coord) -> Pt:
    return (coord["x"], coord["y"]) if isinstance(coord, dict) else (coord.x, coord.y)


def _cells(coords, width: int, height: int) -> list:
    """Converte coordenadas em tuplas. Ponto fora do tabuleiro = estado inválido (cai no fallback)."""
    out = []
    for c in coords or []:
        x, y = _pt(c)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("coordenada fora do tabuleiro")
        out.append((x, y))
    return out


def _body_of(snake, width: int, height: int) -> list:
    """Corpo da cobra, da cabeça à cauda. Se só veio 'head', o corpo é essa única casa."""
    body = _cells(_dig(snake, "body", default=[]), width, height)
    if not body:
        head = _dig(snake, "head")
        if head is not None:
            body = _cells([head], width, height)
    return body


def _latency_ms(you) -> float:
    try:
        return max(0.0, float(_dig(you, "latency", default=0) or 0))
    except (TypeError, ValueError):
        return 0.0


_HISTORY: dict = {}   # id da partida -> [(turno, cabeça)], janela curta (memória do anti-orbitação)


def _track_heads(game_id, turn: int, head: Pt) -> list:
    """Guarda as últimas cabeças desta partida (o servidor é sem estado, então guardamos aqui)."""
    if game_id is None:
        return [head]
    hist = _HISTORY.get(game_id)
    if hist is None:
        if len(_HISTORY) >= 64:
            _HISTORY.pop(next(iter(_HISTORY)))   # esquece a partida mais antiga
        hist = _HISTORY[game_id] = []
    if not hist or hist[-1][0] != turn:           # não conta duas vezes o mesmo turno
        hist.append((turn, head))
    if len(hist) > 14:
        del hist[0]
    return [h for _, h in hist]


def build_context(state: GameState, started: float) -> Context:
    board = state.board
    you = state.you
    my_id = _dig(you, "id")
    width, height = board.width, board.height

    my_body = _body_of(you, width, height)
    bodies = [my_body]
    enemies: list[Enemy] = []
    for snake in _dig(board, "snakes", default=[]):
        if _dig(snake, "id") == my_id:
            continue
        body = _body_of(snake, width, height)
        if not body:
            continue
        bodies.append(body)
        enemies.append(Enemy(
            id=_dig(snake, "id", default=""),
            head=body[0],
            body=body,
            length=len(body),
            health=_dig(snake, "health", default=MAX_HEALTH),
        ))

    ruleset = str(_dig(state, "game", "ruleset", "name", default="standard")).lower()
    settings = _dig(state, "game", "ruleset", "settings")
    hazard_damage = _dig(settings, "hazardDamagePerTurn", default=None)
    if hazard_damage is None:
        hazard_damage = _dig(settings, "hazard_damage_per_turn", default=14)

    timeout_ms = _dig(state, "game", "timeout", default=500)
    if not timeout_ms or timeout_ms <= 0:
        timeout_ms = 500
    constrictor = "constrictor" in ruleset  # cauda nunca sai do lugar, sem comida

    ctx = Context(
        width=width,
        height=height,
        turn=_dig(state, "turn", default=0),
        wrapped="wrapped" in ruleset,       # bordas "dão a volta"
        constrictor=constrictor,
        hazard_damage=int(hazard_damage),
        map_name=str(_dig(state, "game", "map", default="")),
        my_head=my_body[0],
        my_tail=my_body[-1],
        my_body=my_body,
        my_len=len(my_body),
        my_health=_dig(you, "health", default=MAX_HEALTH),
        enemies=enemies,
        food=set(_cells(_dig(board, "food", default=[]), width, height)),
        hazards=set(_cells(_dig(board, "hazards", default=[]), width, height)),
        free_at=_build_free_at(bodies, constrictor),
        deadline=started + (timeout_ms / 1000.0) * TUNING["time_budget"],
        timeout_ms=int(timeout_ms),
        latency=_latency_ms(you),
    )
    ctx.max_enemy_len = max((e.length for e in enemies), default=0)
    ctx.recent = _track_heads(_dig(state, "game", "id"), ctx.turn, ctx.my_head)
    ctx.stall = len(ctx.recent) - len(set(ctx.recent))
    ctx.phase = get_phase(ctx)
    mods = PHASE_MODS.get(ctx.phase, {})
    ctx.w = {k: v * mods.get(k, 1.0) for k, v in WEIGHTS.items()}
    ctx.urgency = health_urgency(ctx)

    for enemy in enemies:
        enemy.next_cells = [c for c in neighbors(ctx, enemy.head) if ctx.free_at.get(c, 0) <= 1]
    ctx.enemy_arrival = _build_enemy_arrival(ctx)
    return ctx


def _build_free_at(bodies: list, constrictor: bool) -> dict:
    """
    Para cada casa ocupada: em que turno (contado após o NOSSO movimento = 1) ela fica livre.

    O segmento i de uma cobra de tamanho L sai do tabuleiro quando 'L - i' movimentos
    acontecem. Logo a cauda (i = L-1) libera no turno 1: pisar nela é seguro.
    Se a cobra acabou de comer, a cauda está duplicada (body[-1] == body[-2]); como pegamos o
    MAIOR valor entre segmentos sobrepostos, essa casa só libera no turno 2 — automático.
    (Quem come NESTE turno também libera a cauda antiga: o motor move a cauda e só depois
    duplica a nova. Por isso não há tratamento especial para "rival colado numa comida".)
    Em 'constrictor' ninguém libera casa nenhuma.
    """
    free_at: dict = {}
    for body in bodies:
        length = len(body)
        for i, seg in enumerate(body):
            t = INF if constrictor else length - i
            if t > free_at.get(seg, 0):
                free_at[seg] = t
    return free_at


def _build_enemy_arrival(ctx: Context) -> dict:
    """Menor turno em que algum adversário chega a cada casa (usado em comida e território)."""
    arrival: dict = {}
    for enemy in ctx.enemies:
        for cell, t in _bfs(ctx, enemy.head, 0).items():
            cur = arrival.get(cell)
            if cur is None or t < cur[0] or (t == cur[0] and enemy.length > cur[1]):
                arrival[cell] = (t, enemy.length)
    return arrival


def get_phase(ctx: Context) -> str:
    if ctx.turn < TUNING["opening_turns"]:
        return "opening"
    if len(ctx.enemies) == 1:
        # Antes todo 1v1 pós-abertura era "endgame" (território x1.3, comida x0.8 o jogo inteiro).
        # Agora só é reta final se estou claramente à frente ou o tabuleiro está cheio.
        e = ctx.enemies[0]
        crowded = (ctx.my_len + e.length) / (ctx.width * ctx.height) >= TUNING["endgame_fill"]
        if ctx.my_len >= e.length + 2 or crowded:
            return "endgame"
    return "midgame"


# --------------------------------------------------------------------------- #
# GEOMETRIA
# --------------------------------------------------------------------------- #

def get_next_position(ctx: Context, pos: Pt, move: str):
    """Casa resultante de um movimento; None se sair do tabuleiro (exceto em 'wrapped')."""
    dx, dy = MOVES[move]
    x, y = pos[0] + dx, pos[1] + dy
    if ctx.wrapped:
        return (x % ctx.width, y % ctx.height)
    if 0 <= x < ctx.width and 0 <= y < ctx.height:
        return (x, y)
    return None


def neighbors(ctx: Context, pos: Pt):
    for move in MOVE_ORDER:
        nxt = get_next_position(ctx, pos, move)
        if nxt is not None:
            yield nxt


def distance(ctx: Context, a: Pt, b: Pt) -> int:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    if ctx.wrapped:
        dx, dy = min(dx, ctx.width - dx), min(dy, ctx.height - dy)
    return dx + dy


def _open_neighbors(ctx: Context, pos: Pt, turn: int) -> int:
    return sum(1 for n in neighbors(ctx, pos) if ctx.free_at.get(n, 0) <= turn)


# --------------------------------------------------------------------------- #
# MOVIMENTOS POSSÍVEIS / SEGURANÇA
# --------------------------------------------------------------------------- #

def health_after_move(ctx: Context, pos: Pt) -> int:
    """Vida após entrar em 'pos'. Comer devolve vida cheia (e não custa vida, ver /rules)."""
    if pos in ctx.food and not ctx.constrictor:
        return MAX_HEALTH
    health = ctx.my_health - 1
    if pos in ctx.hazards:
        health -= ctx.hazard_damage
    return health


def is_position_safe(ctx: Context, pos, turn: int = 1) -> bool:
    """Casa legal: dentro do tabuleiro, livre no turno 'turn' e sem morte por fome/hazard."""
    if pos is None:
        return False
    if ctx.free_at.get(pos, 0) > turn:
        return False
    return health_after_move(ctx, pos) > 0


def get_possible_moves(ctx: Context) -> list[str]:
    moves = []
    for move in MOVE_ORDER:
        if is_position_safe(ctx, get_next_position(ctx, ctx.my_head, move)):
            moves.append(move)
    return moves


def emergency_move(ctx: Context) -> str:
    """
    Nenhum movimento passa nos filtros. Escolhe o "menos pior", de forma determinística:
    1) dentro do tabuleiro, 2) sobrevive à vida, 3) casa que libera mais cedo
    (caso minha estimativa de cauda esteja errada, esse é o que ainda pode dar certo).
    """
    def rank(move: str):
        pos = get_next_position(ctx, ctx.my_head, move)
        if pos is None:
            return (3, INF)
        return (0 if health_after_move(ctx, pos) > 0 else 1, ctx.free_at.get(pos, 0))

    return min(MOVE_ORDER, key=rank)


# --------------------------------------------------------------------------- #
# BFS / FLOOD FILL / DIJKSTRA (com liberação de caudas ao longo do tempo)
# --------------------------------------------------------------------------- #

def _bfs(ctx: Context, start: Pt, start_turn: int) -> dict:
    """BFS: casa -> turno de chegada. Uma casa só é atravessável se já estiver livre nesse turno."""
    arrival = {start: start_turn}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        t = arrival[cell] + 1
        for nxt in neighbors(ctx, cell):
            if nxt in arrival or ctx.free_at.get(nxt, 0) > t:
                continue
            arrival[nxt] = t
            queue.append(nxt)
    return arrival


def flood_fill(ctx: Context, start: Pt, start_turn: int = 1) -> FloodResult:
    """Espaço acessível a partir de 'start' (nossa cabeça após o movimento = turno 1)."""
    arrival = _bfs(ctx, start, start_turn)
    tail_ok = (
        not ctx.constrictor
        and ctx.my_len >= 2
        and ctx.my_tail in arrival
    )
    return FloodResult(arrival=arrival, count=len(arrival), tail_reachable=tail_ok)


def step_cost(ctx: Context, cell: Pt) -> int:
    """Vida gasta ao ENTRAR numa casa: 1 por turno + dano extra se for hazard."""
    return 1 + (ctx.hazard_damage if cell in ctx.hazards else 0)


def dijkstra_routes(ctx: Context, start: Pt, start_turn: int = 1) -> Routes:
    """
    Dijkstra a partir de 'start': casa -> (custo em vida, turno de chegada). O custo é medido em
    PONTOS DE VIDA, então dá para comparar direto com a saúde (um caminho por hazard custa muito
    mais que o número de passos). O turno acompanha o caminho escolhido e serve para respeitar
    a liberação das caudas.
    """
    cost = {start: 0}
    turn = {start: start_turn}
    heap = [(0, start_turn, start)]
    while heap:
        c, t0, cell = heapq.heappop(heap)
        if cost.get(cell) != c or turn.get(cell) != t0:
            continue  # entrada obsoleta
        t = t0 + 1
        for nxt in neighbors(ctx, cell):
            if ctx.free_at.get(nxt, 0) > t:
                continue
            cc = c + step_cost(ctx, nxt)
            cur = cost.get(nxt)
            if cur is None or cc < cur or (cc == cur and t < turn[nxt]):
                cost[nxt] = cc
                turn[nxt] = t
                heapq.heappush(heap, (cc, t, nxt))
    return Routes(cost=cost, turn=turn)


# --------------------------------------------------------------------------- #
# COMPONENTES DE PONTUAÇÃO (cada um devolve pontos: positivo = bom)
# --------------------------------------------------------------------------- #

def evaluate_space(ctx: Context, flood: FloodResult) -> float:
    """Espaço acessível + penalidade de beco (região menor que o corpo)."""
    comfort = max(2 * ctx.my_len, TUNING["space_comfort_min"])
    score = ctx.w["space"] * min(1.0, flood.count / comfort)
    if flood.count < ctx.my_len:
        penalty = ctx.w["dead_end"] * (1.0 - flood.count / ctx.my_len)
        if flood.tail_reachable:  # a cauda vai abrindo espaço: risco menor
            penalty *= 0.5
        score -= penalty
    return score


def evaluate_territory(ctx: Context, flood: FloodResult) -> float:
    """Fatia do tabuleiro que alcançamos antes (ou empatando sendo maiores) dos adversários."""
    mine = 0
    for cell, t in flood.arrival.items():
        enemy = ctx.enemy_arrival.get(cell)
        if enemy is None or t < enemy[0] or (t == enemy[0] and ctx.my_len > enemy[1]):
            mine += 1
    return ctx.w["territory"] * mine / (ctx.width * ctx.height)


def evaluate_mobility(ctx: Context, pos: Pt) -> float:
    """Quantas saídas existirão a partir da nova cabeça (turno 2)."""
    return ctx.w["mobility"] * _open_neighbors(ctx, pos, 2)


def evaluate_tail(ctx: Context, flood: FloodResult) -> float:
    """Cauda alcançável = rota de fuga. Se o espaço está apertado e não há fome, seguir a cauda."""
    if not flood.tail_reachable:
        return 0.0
    score = ctx.w["tail_reach"]
    comfort = max(2 * ctx.my_len, TUNING["space_comfort_min"])
    tightness = max(0.0, 1.0 - flood.count / (1.5 * comfort))  # 0 = folgado, 1 = apertado
    if ctx.my_len >= TUNING["tail_min_len"] or tightness > 0.5:
        d = flood.arrival[ctx.my_tail] - 1
        score += ctx.w["tail_follow"] * tightness * (1.0 - ctx.urgency) / (1 + d)
    return score


def _urgency_at(ctx: Context, hp: float) -> float:
    """0 = vida confortável, 1 = crítica (interpolação linear entre os dois limiares)."""
    lo, hi = TUNING["health_critical"], TUNING["health_comfort"]
    if ctx.hazards:
        lo += TUNING["hazard_urgency_shift"]
        hi += TUNING["hazard_urgency_shift"]
    if hp <= lo:
        return 1.0
    if hp >= hi:
        return 0.0
    return (hi - hp) / (hi - lo)


def health_urgency(ctx: Context) -> float:
    return _urgency_at(ctx, ctx.my_health)


def food_interest(ctx: Context, hp: int, my_len: int, other_len: int, stall: int = 0):
    """
    Quanto a comida vale AGORA, de 0 a 1 (e a urgência de vida, também de 0 a 1).
    interesse = base + (1 - base) * urgência. A base (o que sobra com a vida cheia) sobe na
    abertura, quando estou igual/menor (crescer vence choques) e com pouca comida no mapa;
    cai quando já estou bem maior; e sobe se estou orbitando sem comer.
    """
    need = _urgency_at(ctx, hp)
    if my_len >= other_len + FOOD["ahead_margin"]:
        base = FOOD["base_ahead"]
    else:
        early = max(0.0, 1.0 - ctx.turn / FOOD["early_turns"])
        base = FOOD["base_mid"] + (FOOD["base_open"] - FOOD["base_mid"]) * early
        if my_len <= other_len:
            base += FOOD["behind_bonus"]
        if len(ctx.food) <= TUNING["scarce_food"]:
            base += FOOD["scarce_bonus"]   # (bem maior: não corre atrás de comida escassa)
    if stall >= FOOD["stall_min"]:
        base += FOOD["stall_bonus"]
    base = min(base, 0.85)
    return base + (1.0 - base) * need, need


def evaluate_food(ctx: Context, routes: Routes) -> float:
    """
    Valor da MELHOR comida (não da mais próxima) a partir da nova posição. A distância vem do
    Dijkstra (custo em vida: hazard pesa). Cada comida é descontada se: perderemos a corrida
    (comparada em TURNOS), está em hazard (sem urgência) ou fica em beco. O interesse base sobe
    com a urgência de vida.
    """
    if not ctx.food or ctx.constrictor:
        return 0.0

    best = 0.0
    for f in ctx.food:
        cost = routes.cost.get(f)
        if cost is None:
            continue  # inalcançável a partir daqui
        t = routes.turn[f]
        value = 1.0 / (1 + cost)  # custo 0 = a comida está na casa do movimento

        rival = ctx.enemy_arrival.get(f)
        if rival is not None and (rival[0] < t or (rival[0] == t and rival[1] >= ctx.my_len)):
            value *= TUNING["contested"]
        if f in ctx.hazards and ctx.urgency < 0.8:
            value *= TUNING["hazard_food"]
        if _open_neighbors(ctx, f, t + 1) <= 1:
            value *= TUNING["deadend_food"]
        best = max(best, value)

    ahead = ctx.my_len > ctx.max_enemy_len + 1
    base = TUNING["food_base_ahead"] if ahead else TUNING["food_base"]
    if len(ctx.food) <= TUNING["scarce_food"]:
        base = min(1.0, base + 0.15)
    mix = base + (1.0 - base) * ctx.urgency
    return ctx.w["food"] * mix * best


def evaluate_health(ctx: Context, pos: Pt, routes: Routes) -> float:
    """Penaliza vida baixa e o cenário 'não chego em nenhuma comida a tempo' (custo em vida)."""
    if pos in ctx.food and not ctx.constrictor:
        return 0.0
    h = health_after_move(ctx, pos)
    penalty = 0.0
    low = TUNING["low_health"]
    if h < low:
        penalty += ctx.w["low_health"] * (low - h) / low
    if not ctx.constrictor and ctx.food:
        costs = [routes.cost[f] for f in ctx.food if f in routes.cost]
        if costs and min(costs) >= h:
            penalty += ctx.w["starvation"]
        elif not costs and h <= 30:
            penalty += ctx.w["starvation"] / 2
    return -penalty


def evaluate_hazards(ctx: Context, pos: Pt) -> float:
    """Custo de entrar em hazard; maior quanto menos vida sobraria depois do dano."""
    if pos not in ctx.hazards:
        return 0.0
    if pos in ctx.food:
        return -0.25 * ctx.w["hazard"]  # comer reabastece a vida
    risk = min(1.0, max(0.0, 1.0 - health_after_move(ctx, pos) / MAX_HEALTH))
    return -ctx.w["hazard"] * (1.0 + 2.0 * risk)


def evaluate_head_to_head(ctx: Context, pos: Pt) -> float:
    """
    Para cada adversário que PODE entrar em 'pos' no próximo turno:
      maior  -> perdemos (h2h_loss); igual -> ambos morrem (h2h_tie);
      menor  -> bônus de kill, dividido pelo nº de opções dele (se só tem 1 saída, é quase certo).
    (Se 'pos' tem comida, os dois comeriam e crescem juntos: a comparação de tamanho não muda.)
    """
    score = 0.0
    for e in ctx.enemies:
        if pos not in e.next_cells:
            continue
        if e.length > ctx.my_len:
            score -= ctx.w["h2h_loss"]
        elif e.length == ctx.my_len:
            score -= ctx.w["h2h_tie"]
        else:
            score += ctx.w["kill"] / max(1, len(e.next_cells))
    return score


def evaluate_enemies(ctx: Context, pos: Pt) -> float:
    """Pressão de proximidade: foge de cabeças maiores/iguais, persegue menores se saudável."""
    score = 0.0
    for e in ctx.enemies:
        d = distance(ctx, pos, e.head)
        if e.length >= ctx.my_len:
            if 1 <= d <= TUNING["threat_range"]:
                score -= ctx.w["threat"] / d
        elif d <= TUNING["hunt_range"] and ctx.urgency < 0.5:
            score += ctx.w["hunt"] / max(1, d)
    return score


def evaluate_position(ctx: Context, pos: Pt) -> float:
    """Desempate posicional: centro bom, borda ruim (não se aplica a 'wrapped')."""
    if ctx.wrapped:
        return 0.0
    cx, cy = (ctx.width - 1) / 2, (ctx.height - 1) / 2
    max_d = cx + cy
    center = 1.0 - (abs(pos[0] - cx) + abs(pos[1] - cy)) / max_d if max_d > 0 else 1.0
    score = ctx.w["center"] * center * (1.0 - ctx.urgency)
    if pos[0] in (0, ctx.width - 1) or pos[1] in (0, ctx.height - 1):
        score -= ctx.w["edge"]
    return score


def evaluate_move(ctx: Context, move: str) -> MoveEvaluation:
    """Nota clássica (uma jogada, sem olhar adiante). No 1v1 serve de desempate para a busca."""
    pos = get_next_position(ctx, ctx.my_head, move)
    flood = flood_fill(ctx, pos)
    routes = dijkstra_routes(ctx, pos)
    parts = {
        "space": evaluate_space(ctx, flood),
        "territory": evaluate_territory(ctx, flood),
        "mobility": evaluate_mobility(ctx, pos),
        "tail": evaluate_tail(ctx, flood),
        "food": evaluate_food(ctx, routes),
        "health": evaluate_health(ctx, pos, routes),
        "hazards": evaluate_hazards(ctx, pos),
        "h2h": evaluate_head_to_head(ctx, pos),
        "enemies": evaluate_enemies(ctx, pos),
        "position": evaluate_position(ctx, pos),
    }
    return MoveEvaluation(move=move, pos=pos, score=sum(parts.values()), parts=parts)


def quick_score(ctx: Context, move: str) -> float:
    """Avaliação barata: usada para ordenar a busca e se o orçamento de tempo estourar."""
    pos = get_next_position(ctx, ctx.my_head, move)
    return 10.0 * _open_neighbors(ctx, pos, 2) + evaluate_head_to_head(ctx, pos)


def sort_by_quick_score(ctx: Context, moves: list) -> list:
    """Ordena as jogadas da mais para a menos promissora (estável): melhora a poda alfa-beta."""
    return sorted(moves, key=lambda m: -quick_score(ctx, m))


def classic_choice(ctx: Context, possible: list) -> tuple:
    """Escolha pela pontuação clássica (3+ cobras, ou fallback da busca)."""
    scored = []
    for move in possible:
        if time.perf_counter() > ctx.deadline:
            scored.append((quick_score(ctx, move), move))
            continue
        ev = evaluate_move(ctx, move)
        logger.debug("MOVE %d %s: %.1f %s", ctx.turn, move, ev.score,
                     {k: round(v, 1) for k, v in ev.parts.items()})
        scored.append((ev.score, move))
    best_score = max(s for s, _ in scored)
    best_moves = [m for s, m in scored if s >= best_score - 1e-6]
    return (best_moves[0] if len(best_moves) == 1 else random.choice(best_moves)), best_score


# --------------------------------------------------------------------------- #
# BUSCA 1v1: minimax (alfa-beta) + aprofundamento iterativo + Voronoi em bitboard
# --------------------------------------------------------------------------- #
# Cada lance da árvore é um turno inteiro: eu escolho um movimento e a rival responde com o
# pior para mim (visão pessimista); os dois são aplicados juntos com as regras reais.
#
# O tabuleiro vira um inteiro grande: o bit (y * largura + x) é a casa (x, y). Expandir a
# fronteira do flood fill é um punhado de deslocamentos e ANDs em bloco, bem mais rápido em
# Python do que percorrer casa por casa.

try:
    _popcount = int.bit_count  # Python 3.10+
except AttributeError:  # pragma: no cover
    def _popcount(x: int) -> int:
        return bin(x).count("1")


class _SearchTimeout(Exception):
    """Estourou o prazo (é só um sinal de controle)."""


@lru_cache(maxsize=16)
def _geometry(width: int, height: int, wrapped: bool):
    """Tabelas que só dependem do tamanho do tabuleiro: vizinhos, bits e expansão de fronteira."""
    V = width * height
    nbr = []
    for c in range(V):
        x, y = c % width, c // width
        row = []
        for dx, dy in (MOVES[m] for m in MOVE_ORDER):
            nx, ny = x + dx, y + dy
            if wrapped:
                row.append((ny % height) * width + (nx % width))
            elif 0 <= nx < width and 0 <= ny < height:
                row.append(ny * width + nx)
            else:
                row.append(-1)
        nbr.append(tuple(row))
    bit = [1 << i for i in range(V)]
    full = (1 << V) - 1
    col0 = 0
    colL = 0
    for y in range(height):
        col0 |= 1 << (y * width)
        colL |= 1 << (y * width + width - 1)
    not_l = full & ~col0   # células que NÃO estão na coluna x = 0
    not_r = full & ~colL   # células que NÃO estão na coluna x = largura-1
    W, shift_v = width, V - width

    if not wrapped:
        def expand(f: int) -> int:
            return (((f & not_r) << 1) | ((f & not_l) >> 1) | (f << W) | (f >> W)) & full
    else:
        def expand(f: int) -> int:
            return (((f & not_r) << 1) | ((f & colL) >> (W - 1))
                    | ((f & not_l) >> 1) | ((f & col0) << (W - 1))
                    | (f << W) | (f >> shift_v)
                    | (f >> W) | (f << shift_v)) & full

    return tuple(nbr), bit, full, expand


class _St:
    """Estado dentro da busca (cobra 0 = eu, cobra 1 = rival). Não é alterado depois de criado."""
    __slots__ = ("b0", "b1", "h0", "h1", "food", "pm0", "pm1", "occ")


class Search:
    def __init__(self, ctx: Context, rival: Enemy):
        self.W, self.H = ctx.width, ctx.height
        self.V = self.W * self.H
        self.wrapped = ctx.wrapped
        self.constrictor = ctx.constrictor
        self.hdmg = ctx.hazard_damage
        self.nbr, self.bit, self.full, self.expand = _geometry(self.W, self.H, self.wrapped)
        self.hazard = [False] * self.V
        for (x, y) in ctx.hazards:
            self.hazard[y * self.W + x] = True
        self.any_hazard = bool(ctx.hazards)
        self.rival = rival
        self.my_len0 = ctx.my_len
        self.deadline = 0.0
        self.nodes = 0
        self.max_depth = 0
        self.res_moves: list = []
        self.res_vals: list = []
        self._prepare(ctx)

    # ------------------------------------------------- preparação (1x por jogada)

    def _prepare(self, ctx: Context) -> None:
        """
        Tudo que as folhas precisam saber sobre comida e fase, calculado UMA vez por jogada.
        Dentro de evaluate() só há consultas a listas e contas simples (nada de varrer o tabuleiro).
        """
        rival = self.rival
        self.root_l0, self.root_l1 = ctx.my_len, rival.length
        self.terr = ctx.w["territory"] / WEIGHTS["territory"]   # fase: reta final = mais território
        self.press = ctx.w["hunt"] / WEIGHTS["hunt"]            # fase: abertura 0.3x, reta final 2x
        peak = ctx.w["s_food"]                                  # já multiplicado por PHASE_MODS
        self.mix0, need0 = food_interest(ctx, ctx.my_health, ctx.my_len, rival.length, ctx.stall)
        mix1, need1 = food_interest(ctx, rival.health, rival.length, ctx.my_len)
        self.fs0 = peak * self.mix0                 # valor de uma comida segura colada em mim
        self.fs1 = peak * mix1 * FOOD["deny"]       # quanto me custa o rival levar uma comida
        span = FOOD["h_urgent"] - FOOD["h_min"]
        self.inv0 = 1.0 / (FOOD["h_min"] + span * need0)   # 1/alcance: com fome, comida longe importa
        self.inv1 = 1.0 / (FOOD["h_min"] + span * need1)
        self.fq = [0.0] * self.V                    # qualidade (0..1.5) de cada comida atual
        self.eat0, self.eat1 = {}, {}               # recompensa de comer (eu / rival), por casa
        self.root_food = 0
        self.commit = None                          # comida comprometida (Pt) ou None
        self.commit_dist = None                     # distância (passos) de cada casa até ela
        self.commit_d0 = 0                          # minha distância atual até ela
        if ctx.food and not self.constrictor:
            self._food_quality(ctx, need0)
            for p in ctx.food:
                idx = self._cell(p)
                self.root_food |= self.bit[idx]
                # comer vale um pouco MAIS que ficar colado na comida: sem isso a busca orbitaria
                self.eat0[idx] = self.fs0 * (self.fq[idx] + FOOD["eat_extra"])
                self.eat1[idx] = self.fs1 * (self.fq[idx] + FOOD["eat_extra"])

    def _food_quality(self, ctx: Context, need0: float) -> None:
        """
        Qualidade de cada comida (1.0 = ótima; 0 = não vale/alcanço):
          - não chego com a vida que tenho (custo em vida via Dijkstra se há hazard) -> 0
          - em hazard sem necessidade, em corredor/canto, com pouco espaço depois de comer,
            custando quase toda a vida, ou com rival MAIOR chegando logo atrás -> desconto
        E escolhe a comida comprometida: segura, chego antes do rival e vale a pena.
        Custo: 1 BFS da cabeça (+1 Dijkstra com hazard) + 1 BFS por comida próxima (até 4) + 1 BFS
        da comida comprometida. Roda uma vez por jogada, não por folha.
        """
        d_me = _bfs(ctx, ctx.my_head, 0)
        routes = dijkstra_routes(ctx, ctx.my_head, 0) if ctx.hazards else None
        hp = ctx.my_health
        reach = []
        for f in ctx.food:
            d = d_me.get(f)
            if d is None:
                continue
            cost = routes.cost.get(f) if routes is not None else d   # custo em PONTOS DE VIDA
            if cost is None:
                continue
            if f in ctx.hazards:
                cost -= ctx.hazard_damage     # comer repõe a vida: o dano da própria casa não conta
            if cost >= hp:
                continue                      # morreria de fome a caminho
            reach.append((d, cost, f))
        reach.sort()

        fq = self.fq
        best_val, best = 0.0, None
        for rank, (d, cost, f) in enumerate(reach):
            q = 1.0
            if f in ctx.hazards:
                q *= TUNING["hazard_food"] + (1.0 - TUNING["hazard_food"]) * min(1.0, need0 / 0.8)
            if routes is not None and cost > 0.6 * hp and need0 < 0.6:
                q *= 0.5                      # gastaria quase toda a vida sem precisar
            if _open_neighbors(ctx, f, d + 1) <= 1:
                q *= TUNING["deadend_food"]   # canto/corredor
            if rank < FOOD["space_checks"]:
                space = len(_bfs(ctx, f, d))  # região acessível DEPOIS de comer (caudas liberando)
                room = ctx.my_len + 1
                if space < room:
                    q *= 0.3
                elif space < 2 * room:
                    q *= 0.75
            rt, rl = ctx.enemy_arrival.get(f, (None, 0))
            margin = (rt - d) if rt is not None else 99     # >0: chego antes do rival
            if margin == 1 and rl > ctx.my_len:
                q *= 0.6                      # rival maior logo atrás: head-to-head ruim
            fq[self._cell(f)] = q
            if (margin > 0 or (margin == 0 and ctx.my_len > rl)) and q >= 0.5:
                val = self.mix0 * q * max(0.0, 1.0 - (d - 1) * self.inv0)
                if val > best_val:
                    best_val, best = val, (f, d)

        if best is not None and best_val >= FOOD["commit_min"]:
            f, d = best
            fq[self._cell(f)] *= FOOD["commit_boost"]
            self.commit, self.commit_d0 = f, d
            self.commit_dist = _bfs(ctx, f, 0)

    # ------------------------------------------------------------------ estado

    def _cell(self, p: Pt) -> int:
        return p[1] * self.W + p[0]

    def _new(self, b0: tuple, b1: tuple, h0: int, h1: int, food: int) -> _St:
        s = _St()
        s.b0, s.b1, s.h0, s.h1, s.food = b0, b1, h0, h1, food
        bit = self.bit
        pm0 = [0]
        m = 0
        for c in b0:
            m |= bit[c]
            pm0.append(m)
        pm1 = [0]
        m = 0
        for c in b1:
            m |= bit[c]
            pm1.append(m)
        s.pm0, s.pm1 = pm0, pm1
        # casas ocupadas para colisão: corpo sem a cauda (que sai); em constrictor, corpo inteiro
        s.occ = (pm0[-1] | pm1[-1]) if self.constrictor else (pm0[len(b0) - 1] | pm1[len(b1) - 1])
        return s

    def make_state(self, ctx: Context) -> _St:
        food = 0
        for p in ctx.food:
            food |= self.bit[self._cell(p)]
        return self._new(
            tuple(self._cell(p) for p in ctx.my_body),
            tuple(self._cell(p) for p in self.rival.body),
            ctx.my_health, self.rival.health, food,
        )

    # ------------------------------------------------------------------ regras

    def step(self, s: _St, m0: int, m1: int):
        """Um turno com os dois movimentos. Devolve (novo estado ou None, morri, rival morreu)."""
        nbr, bit = self.nbr, self.bit
        n0 = nbr[s.b0[0]][m0]
        n1 = nbr[s.b1[0]][m1]
        food = s.food
        if self.constrictor:
            nh0, nh1 = s.h0, s.h1
            e0 = e1 = False
            g0 = g1 = 1
        else:
            nh0, nh1 = s.h0 - 1, s.h1 - 1
            e0 = n0 >= 0 and bool(food & bit[n0])
            e1 = n1 >= 0 and bool(food & bit[n1])
            if n0 >= 0 and self.hazard[n0]:
                nh0 -= self.hdmg
            if n1 >= 0 and self.hazard[n1]:
                nh1 -= self.hdmg
            if e0:
                nh0 = MAX_HEALTH
            if e1:
                nh1 = MAX_HEALTH
            g0, g1 = int(e0), int(e1)
        d0 = n0 < 0 or nh0 <= 0
        d1 = n1 < 0 or nh1 <= 0
        occ = s.occ
        if not d0 and occ & bit[n0]:
            d0 = True
        if not d1 and occ & bit[n1]:
            d1 = True
        if n0 == n1 and n0 >= 0:  # choque de cabeças: o menor morre, igual morrem os dois
            l0 = len(s.b0) + g0
            l1 = len(s.b1) + g1
            if l0 <= l1:
                d0 = True
            if l1 <= l0:
                d1 = True
        if d0 or d1:
            return None, d0, d1
        if self.constrictor:
            nb0 = (n0,) + s.b0
            nb1 = (n1,) + s.b1
        else:
            # o motor primeiro move (a cauda sai) e depois, se comeu, duplica a NOVA cauda
            nb0 = (n0,) + s.b0[:-1]
            nb1 = (n1,) + s.b1[:-1]
            if e0:
                nb0 += (nb0[-1],)
            if e1:
                nb1 += (nb1[-1],)
            if e0 or e1:
                if e0:
                    food &= ~bit[n0]
                if e1:
                    food &= ~bit[n1]
        return self._new(nb0, nb1, nh0, nh1, food), False, False

    def moves(self, s: _St, who: int) -> list:
        """Movimentos que não matam de imediato (parede/corpo; caudas que saem contam como livres)."""
        head = (s.b0 if who == 0 else s.b1)[0]
        occ, bit, row = s.occ, self.bit, self.nbr[head]
        out = [m for m in range(4) if row[m] >= 0 and not (occ & bit[row[m]])]
        return out or [0]  # sem saída: um qualquer (a simulação marca como morte)

    # ------------------------------------------------------------- avaliação

    def _blocked(self, s: _St, t: int) -> int:
        """Casas ainda ocupadas no passo t (segmento i de cobra de tamanho L sai no passo L - i)."""
        if self.constrictor:
            return s.pm0[-1] | s.pm1[-1]
        k0 = len(s.b0) - t
        k1 = len(s.b1) - t
        return (s.pm0[k0] if k0 > 0 else 0) | (s.pm1[k1] if k1 > 0 else 0)

    def _food_dist(self, s: _St, head: int, limit: int) -> int:
        """Menor distância (em turnos) até alguma comida, ignorando o rival. -1 se não houver."""
        seen = front = self.bit[head]
        expand, full, food = self.expand, self.full, s.food
        t = 0
        while front and t < limit:
            t += 1
            nxt = expand(front) & full & ~(seen | self._blocked(s, t))
            if nxt & food:
                return t
            seen |= nxt
            front = nxt
        return -1

    def _food_life_cost(self, s: _St, head: int, hp: int) -> int:
        """
        Custo em PONTOS DE VIDA até a comida mais barata (hazard custa 1 + dano por casa). Só é
        usado quando há hazards e a vida está baixa. -1 se não alcança com a vida que tem.
        """
        free_t: dict = {}
        for body in (s.b0, s.b1):
            L = len(body)
            for i, c in enumerate(body):
                if L - i > free_t.get(c, 0):
                    free_t[c] = L - i
        bit, food = self.bit, s.food
        best = {head: 0}
        heap = [(0, 0, head)]
        while heap:
            cost, steps, c = heapq.heappop(heap)
            if cost > best.get(c, INF):
                continue
            if c != head and food & bit[c]:
                return cost - (self.hdmg if self.hazard[c] else 0)
            if cost >= hp:
                continue
            for n in self.nbr[c]:
                if n < 0 or free_t.get(n, 0) > steps + 1:
                    continue
                nc = cost + 1 + (self.hdmg if self.hazard[n] else 0)
                if nc < best.get(n, INF):
                    best[n] = nc
                    heapq.heappush(heap, (nc, steps + 1, n))
        return -1

    def evaluate(self, s: _St) -> float:
        """Nota do estado do ponto de vista da cobra 0 (eu): positivo = bom para mim."""
        b0, b1 = s.b0, s.b1
        l0, l1 = len(b0), len(b1)
        pm0, pm1 = s.pm0, s.pm1
        bit, full, expand = self.bit, self.full, self.expand
        food = s.food
        constrictor = self.constrictor
        pc = _popcount
        fq = self.fq
        # depois que EU comi, as outras comidas valem menos (vida cheia); idem para o rival
        fs0 = self.fs0 * (0.5 if l0 > self.root_l0 else 1.0)
        fs1 = self.fs1 * (0.5 if l1 > self.root_l1 else 1.0)
        inv0, inv1 = self.inv0, self.inv1

        h0b, h1b = bit[b0[0]], bit[b1[0]]
        claimed = h0b | h1b
        f0, f1 = h0b, h1b
        c0 = c1 = 0
        food_sc = 0.0
        t = 0
        # Voronoi: a cada passo cada cobra expande a fronteira; casa alcançada por uma só é dela;
        # disputada (mesmo turno) fica com a MAIOR (a menor perderia o choque); igual = de ninguém.
        while f0 or f1:
            t += 1
            if constrictor:
                blocked = pm0[l0] | pm1[l1]
            else:
                k0, k1 = l0 - t, l1 - t
                blocked = (pm0[k0] if k0 > 0 else 0) | (pm1[k1] if k1 > 0 else 0)
            avail = full & ~(claimed | blocked)
            n0 = expand(f0) & avail
            n1 = expand(f1) & avail
            both = n0 & n1
            if both:
                if l0 > l1:
                    m0b, m1b = n0, n1 & ~both
                elif l1 > l0:
                    m0b, m1b = n0 & ~both, n1
                else:
                    m0b, m1b = n0 & ~both, n1 & ~both
            else:
                m0b, m1b = n0, n1
            c0 += pc(m0b)
            c1 += pc(m1b)
            if food:
                # comida que chego primeiro: valor = escala dinâmica x qualidade x proximidade
                h0f = m0b & food
                if h0f:
                    g = 1.0 - (t - 1) * inv0
                    if g > 0.0:
                        k = fs0 * g
                        while h0f:
                            low = h0f & -h0f
                            food_sc += k * fq[low.bit_length() - 1]
                            h0f ^= low
                h1f = m1b & food
                if h1f:
                    g = 1.0 - (t - 1) * inv1
                    if g > 0.0:
                        k = fs1 * g
                        while h1f:
                            low = h1f & -h1f
                            food_sc -= k * fq[low.bit_length() - 1]
                            h1f ^= low
            claimed |= n0 | n1
            f0, f1 = n0, n1

        sc = self.terr * (c0 - c1) + SEARCH["w_length"] * (l0 - l1) + food_sc
        if self.root_food:
            # comida que SUMIU durante a linha foi comida: recompensa quem cresceu (valor médio)
            eaten = self.root_food & ~food
            if eaten:
                g0, g1 = l0 - self.root_l0, l1 - self.root_l1
                if g0 > 0 or g1 > 0:
                    e0 = e1 = 0.0
                    n = 0
                    while eaten:
                        low = eaten & -eaten
                        idx = low.bit_length() - 1
                        e0 += self.eat0[idx]
                        e1 += self.eat1[idx]
                        n += 1
                        eaten ^= low
                    if g0 > 0:
                        sc += g0 * e0 / n
                    if g1 > 0:
                        sc -= g1 * e1 / n
        if c0 < l0:
            sc -= SEARCH["w_cramped"] * (l0 - c0)
        if c1 < l1:
            sc += SEARCH["w_cramped"] * self.press * (l1 - c1)   # pressão: abertura fraca, reta final forte

        if not constrictor:
            for who in (0, 1):
                hp = s.h0 if who == 0 else s.h1
                if hp >= 45:
                    continue
                head = b0[0] if who == 0 else b1[0]
                if food and self.any_hazard:
                    d = self._food_life_cost(s, head, hp)
                elif food:
                    d = self._food_dist(s, head, hp)
                else:
                    d = -1
                if d < 0:
                    # há comida mas não alcanço (ou não há comida e a vida está no fim)
                    pen = SEARCH["w_starve"] * (45 - hp) / 45 if food else (
                        SEARCH["w_starve"] if hp <= 5 else 0.0)
                else:
                    margin = hp - d
                    pen = SEARCH["w_starve"] if margin <= 0 else (
                        SEARCH["w_starve_margin"] * max(0, 8 - margin))
                sc += -pen if who == 0 else pen
        if self.any_hazard:
            if self.hazard[b0[0]]:
                sc -= SEARCH["w_hazard"]
            if self.hazard[b1[0]]:
                sc += SEARCH["w_hazard"]
        return sc

    # ----------------------------------------------------------------- busca

    def _tick(self) -> None:
        self.nodes += 1
        if (self.nodes & 7) == 0 and time.perf_counter() > self.deadline:
            raise _SearchTimeout()

    def _child(self, s, m0, m1, depth, alpha, beta, ply) -> float:
        ns, d0, d1 = self.step(s, m0, m1)
        if d0:
            return DRAW if d1 else -(WIN - ply)   # os dois morrem = empate (ruim, mas não é derrota)
        if d1:
            return WIN - ply                     # quanto mais cedo a vitória, melhor
        return self._ab(ns, depth - 1, alpha, beta, ply + 1)

    def _ab(self, s: _St, depth: int, alpha: float, beta: float, ply: int) -> float:
        self._tick()
        if depth <= 0:
            return self.evaluate(s)
        mine = self.moves(s, 0)
        theirs = self.moves(s, 1)
        best = -10.0 * WIN
        for m0 in mine:
            worst = 10.0 * WIN
            for m1 in theirs:
                v = self._child(s, m0, m1, depth, alpha, min(beta, worst), ply)
                if v < worst:
                    worst = v
                if worst <= alpha:
                    break
            if worst > best:
                best = worst
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def search(self, s: _St, order: list, deadline: float) -> None:
        """Aprofundamento iterativo. O resultado da última profundidade COMPLETA fica em res_*."""
        self.deadline = deadline
        self.res_moves, self.res_vals = [], []
        theirs = self.moves(s, 1)
        for depth in range(1, 60):
            mv, val = [], []
            alpha = -10.0 * WIN
            try:
                for m0 in order:
                    worst = 10.0 * WIN
                    for m1 in theirs:
                        # folga ROOT_EPS: jogadas de valor IGUAL ao melhor saem exatas (sem falso empate)
                        v = self._child(s, m0, m1, depth, alpha - ROOT_EPS, worst, 0)
                        if v < worst:
                            worst = v
                        if worst <= alpha - ROOT_EPS:
                            break
                    mv.append(m0)
                    val.append(worst)
                    if worst > alpha:
                        alpha = worst
            except _SearchTimeout:
                break
            self.res_moves, self.res_vals = mv, val
            self.max_depth = depth
            # próxima profundidade: melhores primeiro (ordenação estável por valor decrescente)
            idx = sorted(range(len(mv)), key=lambda i: -val[i])
            order = [mv[i] for i in idx]
            mx = max(val)
            if mx >= WIN - 200 or mx <= -(WIN - 200):
                break   # vitória ou derrota forçada: aprofundar não muda nada


def search_budget_ms(ctx: Context) -> float:
    """
    Tempo da busca. Padrão: 30% do timeout (máx. cap_ms). Se a latência medida pelo jogo na
    jogada anterior (rede + nosso cálculo) já passa de 60% do timeout, encolhe a busca na mesma
    medida para a resposta não estourar o limite.
    """
    base = min(SEARCH["cap_ms"], ctx.timeout_ms * SEARCH["time"])
    over = max(0.0, ctx.latency - 0.6 * ctx.timeout_ms)
    return max(SEARCH["min_ms"], base - over)


def _terminal(v: float) -> bool:
    """Vitória/derrota forçada ou empate por morte mútua: esses valores nunca são ajustados."""
    return abs(v) >= WIN - 200 or v == DRAW


def _root_adjustments(ctx: Context, se: "Search", moves: list):
    """
    Bônus de PROGRESSO por jogada, na raiz (pequeno e limitado):
      - rumo à comida comprometida: +1 se aproxima, -1 se afasta, 0 se lateral. É um potencial
        (diferença de distância), então ir e voltar soma zero: não dá para "farmar" repetição.
      - se estou orbitando (posições repetidas), revisitar uma casa recente custa um pouco.
    Devolve ({jogada: bônus}, teto de diferença de valor em que o bônus pode decidir).
    """
    pts = ctx.w["s_commit"] * (0.35 + 0.65 * se.mix0)
    out = {}
    for m in moves:
        pos = get_next_position(ctx, ctx.my_head, MOVE_ORDER[m])
        bonus = 0.0
        if se.commit_dist is not None and pos is not None:
            da = se.commit_dist.get(pos)
            progress = -1 if da is None else max(-1, min(1, se.commit_d0 - da))
            bonus += pts * progress
        if ctx.stall >= FOOD["stall_min"] and pos is not None:
            bonus -= FOOD["revisit_pen"] * min(3, ctx.recent[:-1].count(pos))
        out[m] = bonus
    return out, 2.0 * pts + 3.0 * FOOD["revisit_pen"]


def _search_choice(ctx: Context, possible: list, started: float):
    """Escolha por busca (1v1). Devolve (movimento, descrição) ou (None, '') se a busca não rendeu."""
    deadline = started + search_budget_ms(ctx) / 1000.0
    order = [MOVE_ORDER.index(m) for m in sort_by_quick_score(ctx, possible)]
    se = Search(ctx, ctx.enemies[0])
    se.search(se.make_state(ctx), order, deadline)
    if not se.res_moves:
        return None, ""
    best_v = max(se.res_vals)
    # Jogadas de valor parecido (dentro do teto) são decididas pelo progresso estratégico real.
    # Vitória/derrota forçada e empate por morte mútua nunca são mexidos: sobrevivência manda.
    adj, cap = ({}, 0.0) if _terminal(best_v) else _root_adjustments(ctx, se, se.res_moves)
    scored = []
    for m, v in zip(se.res_moves, se.res_vals):
        bonus = adj.get(m, 0.0) if (v >= best_v - cap and not _terminal(v)) else 0.0
        scored.append((v + bonus, m))
    best_sc = max(sc for sc, _ in scored)
    tied = [m for sc, m in scored if sc >= best_sc - 1e-9]
    pick = tied[0]
    if len(tied) > 1:  # empate: a pontuação clássica decide (se ainda houver tempo)
        guard = started + 0.40 * ctx.timeout_ms / 1000.0
        best_key = None
        for m in tied:
            name = MOVE_ORDER[m]
            key = evaluate_move(ctx, name).score if time.perf_counter() < guard else quick_score(ctx, name)
            if best_key is None or key > best_key:
                pick, best_key = m, key
    info = "busca d=%d nós=%d v=%.0f" % (se.max_depth, se.nodes, best_v)
    if se.commit is not None:
        info += " comida=%s" % (se.commit,)
    return MOVE_ORDER[pick], info


# --------------------------------------------------------------------------- #
# FALLBACK — só roda se a lógica principal lançar exceção
# --------------------------------------------------------------------------- #

def _legacy_safe_moves(state: GameState) -> list[str]:
    """Paredes + todos os corpos (nossos e dos rivais). Respeita o modo 'wrapped'."""
    wrapped = "wrapped" in str(_dig(state, "game", "ruleset", "name", default="")).lower()
    w, h = state.board.width, state.board.height
    occupied = {_pt(c) for c in _dig(state.you, "body", default=[])}
    for snake in _dig(state.board, "snakes", default=[]):
        occupied |= {_pt(c) for c in _dig(snake, "body", default=[])}
    body = _dig(state.you, "body", default=[])
    head = _pt(body[0] if body else state.you.head)
    safe = []
    for name, (dx, dy) in MOVES.items():
        x, y = head[0] + dx, head[1] + dy
        if wrapped:
            x, y = x % w, y % h
        if 0 <= x < w and 0 <= y < h and (x, y) not in occupied:
            safe.append(name)
    return safe


# --------------------------------------------------------------------------- #
# PONTO DE ENTRADA
# --------------------------------------------------------------------------- #

def get_move(state: GameState) -> MoveResponse:
    started = time.perf_counter()
    try:
        ctx = build_context(state, started)
        possible = get_possible_moves(ctx)

        if not possible:
            move = emergency_move(ctx)
            logger.info("MOVE %d: sem saída! emergência -> %s", state.turn, move)
            return MoveResponse(move=move)
        if len(possible) == 1:  # jogada única: nem precisa pensar
            return MoveResponse(move=possible[0])

        chosen, how = None, "pontos"
        if USE_SEARCH and len(ctx.enemies) == 1:  # 1v1: busca; com 3+ cobras vale a pontuação
            try:
                chosen, how = _search_choice(ctx, possible, started)
            except Exception:
                logger.exception("MOVE %s: erro na busca, usando pontuação", state.turn)
                chosen = None
        if chosen is None:
            chosen, score = classic_choice(ctx, possible)
            how = "pontos %.1f" % score

        logger.info("MOVE %d [%s]: %s (%s, %.0f ms)", state.turn, ctx.phase, chosen, how,
                    (time.perf_counter() - started) * 1000)
        return MoveResponse(move=chosen)

    except Exception:  # nunca devolver erro HTTP: um movimento ruim é melhor que nenhum
        logger.exception("MOVE %s: erro na lógica, usando fallback", getattr(state, "turn", "?"))
        try:
            safe = _legacy_safe_moves(state)
        except Exception:
            safe = []
        return MoveResponse(move=safe[0] if safe else "up")