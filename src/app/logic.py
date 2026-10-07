# Bem-vindo ao
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \   __\   __\  | _/ __ \ /  ___//    \__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\_____>
#
# =============================================================================
#  v2.0.0 — O QUE MUDOU EM RELAÇÃO À v1.1.0
# =============================================================================
#  1. BUSCA 1v1 (classe Search): minimax com poda alfa-beta e aprofundamento iterativo.
#     Cada lance da árvore é um turno inteiro (eu + rival juntos, regras reais: comida,
#     fome, hazard, parede, corpo, choque de cabeças). O rival joga o pior caso para mim.
#     Com 3+ cobras continua a pontuação por componentes (a v1.1.0 inteira).
#  2. AVALIAÇÃO DA BUSCA EM BITBOARD: o Voronoi (quem chega primeiro em cada casa) é feito
#     com inteiros grandes do Python (operações em bloco, sem laços por casa). Isso torna a
#     avaliação ~10x mais barata e é o que permite profundidade útil em Python puro.
#  3. DIJKSTRA EM PONTOS DE VIDA para comida/fome na pontuação clássica: em mapas com
#     hazard, cada casa de hazard custa 1 + dano (antes só se contavam passos).
#  4. ORÇAMENTO DE TEMPO ADAPTATIVO: a busca encolhe se you.latency (rede + cálculo da
#     jogada anterior) já estiver perto do timeout.
#  5. DESEMPATE NA RAIZ SEM "FALSO EMPATE": a janela da raiz usa uma folga mínima para que
#     valores iguais sejam exatos (alfa-beta puro devolve só um limite em jogadas cortadas).
#  6. JOGADA ÚNICA = RESPOSTA IMEDIATA (não gasta tempo avaliando).
#  7. ROBUSTEZ: valida coordenadas e o fallback agora considera TODOS os corpos e 'wrapped'.
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
}

# Busca 1v1. Pesos da avaliação da busca: 1 ponto = 1 casa de território de vantagem.
SEARCH: dict[str, float] = {
    "time": 0.30,           # fração do timeout usada pela busca
    "cap_ms": 220.0,        # teto de tempo da busca
    "min_ms": 25.0,         # piso de tempo (mesmo com latência alta)
    "w_length": 3.0,        # por segmento a mais que a rival (ganha choques e território)
    "w_food": 7.0,          # comida que chego antes do rival (decai com a distância)
    "w_food_lost": 2.5,     # comida que o rival chega antes
    "w_starve": 400.0,      # morro de fome antes de alcançar qualquer comida
    "w_starve_margin": 6.0, # por turno de folga abaixo de 8 até a comida mais próxima
    "w_cramped": 6.0,       # por casa que falta para o território igualar meu tamanho
    "w_hazard": 20.0,       # cabeça dentro de hazard
}
WIN = 100000
ROOT_EPS = 1e-6

# Multiplicadores por fase da partida (só altera o que for listado).
PHASE_MODS: dict[str, dict[str, float]] = {
    "opening": {"food": 1.3, "hunt": 0.3, "kill": 0.7},
    "midgame": {},
    "endgame": {"hunt": 2.0, "territory": 1.3, "food": 0.8},  # 1v1: pressionar o adversário
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
        "version": "2.0.0",
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


def health_urgency(ctx: Context) -> float:
    """0 = vida confortável, 1 = crítica (interpolação linear entre os dois limiares)."""
    lo, hi = TUNING["health_critical"], TUNING["health_comfort"]
    if ctx.hazards:
        lo += TUNING["hazard_urgency_shift"]
        hi += TUNING["hazard_urgency_shift"]
    h = ctx.my_health
    if h <= lo:
        return 1.0
    if h >= hi:
        return 0.0
    return (hi - h) / (hi - lo)


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
        W_FOOD, W_FOOD_LOST = SEARCH["w_food"], SEARCH["w_food_lost"]

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
                a = pc(m0b & food)
                b = pc(m1b & food)
                if a:
                    food_sc += W_FOOD * a / (1 + t)
                if b:
                    food_sc -= W_FOOD_LOST * b / (1 + t)
            claimed |= n0 | n1
            f0, f1 = n0, n1

        sc = (c0 - c1) + SEARCH["w_length"] * (l0 - l1) + food_sc
        if c0 < l0:
            sc -= SEARCH["w_cramped"] * (l0 - c0)
        if c1 < l1:
            sc += SEARCH["w_cramped"] * (l1 - c1)

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
            return 0.0 if d1 else -(WIN - ply)   # os dois morrem = empate
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


def _search_choice(ctx: Context, possible: list, started: float):
    """Escolha por busca (1v1). Devolve (movimento, descrição) ou (None, '') se a busca não rendeu."""
    deadline = started + search_budget_ms(ctx) / 1000.0
    order = [MOVE_ORDER.index(m) for m in sort_by_quick_score(ctx, possible)]
    se = Search(ctx, ctx.enemies[0])
    se.search(se.make_state(ctx), order, deadline)
    if not se.res_moves:
        return None, ""
    best_v = max(se.res_vals)
    tied = [m for m, v in zip(se.res_moves, se.res_vals) if v >= best_v - 1e-9]
    pick = tied[0]
    if len(tied) > 1:  # empate: a pontuação clássica decide (se ainda houver tempo)
        guard = started + 0.40 * ctx.timeout_ms / 1000.0
        best_key = None
        for m in tied:
            name = MOVE_ORDER[m]
            key = evaluate_move(ctx, name).score if time.perf_counter() < guard else quick_score(ctx, name)
            if best_key is None or key > best_key:
                pick, best_key = m, key
    return MOVE_ORDER[pick], "busca d=%d nós=%d v=%.0f" % (se.max_depth, se.nodes, best_v)


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