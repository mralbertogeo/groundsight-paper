-- =====================================================================
-- Esquema relacional — dados SIAGAS e derivados
-- Convenção: dados brutos imutáveis (raw_*), derivados recalculáveis.
-- =====================================================================
CREATE EXTENSION IF NOT EXISTS postgis;

-- Cadastro (Etapa 1)
CREATE TABLE pocos (
    codigo_siagas   text PRIMARY KEY,
    geom            geometry(Point, 4674) NOT NULL,     -- SIRGAS 2000
    cota_terreno_m  numeric,
    cota_fonte      text,                                -- 'siagas' | 'mde'
    profundidade_m  numeric,
    natureza        text,
    data_perfuracao date,
    uso             text,
    qaqc_aprovado   boolean DEFAULT false,
    qaqc_flags      text[] DEFAULT '{}'                  -- regras não bloqueantes
);
CREATE INDEX pocos_geom_idx ON pocos USING gist (geom);

-- Perfil litológico (Etapa 1)
CREATE TABLE litologia (
    codigo_siagas  text REFERENCES pocos ON DELETE CASCADE,
    de_m           numeric NOT NULL,
    ate_m          numeric NOT NULL,
    descricao_raw  text,                                 -- texto livre SIAGAS
    hidrolito      text,                                 -- classe do dicionário de-para
    CHECK (ate_m > de_m)
);

-- Perfil construtivo: seções filtrantes (Etapa 1) — campo crítico
CREATE TABLE filtros (
    codigo_siagas  text REFERENCES pocos ON DELETE CASCADE,
    de_m           numeric NOT NULL,
    ate_m          numeric NOT NULL,
    CHECK (ate_m > de_m)
);

-- Teste de bombeamento bruto (Etapa 1)
CREATE TABLE hidraulica_raw (
    codigo_siagas  text REFERENCES pocos ON DELETE CASCADE,
    vazao_m3h      numeric,
    ne_m           numeric,
    nd_m           numeric,
    tempo_h        numeric,
    metodo         text
);

-- Derivados hidráulicos (Etapa 3)
CREATE TABLE hidraulica_derivada (
    codigo_siagas   text PRIMARY KEY REFERENCES pocos,
    qs_m3h_m        numeric,     -- capacidade específica
    log_qs          numeric,
    t_m2_d          numeric,     -- transmissividade estimada (relação declarada)
    relacao_t_qs    text,        -- ex.: 'mace_1997'
    k_m_d           numeric,     -- K = T / L_f
    lf_m            numeric,     -- comprimento total de filtros
    bsat_m          numeric,
    carga_h_m       numeric,     -- cota - NE
    unidade_captada text,        -- hidroestratigrafia dominante (Etapa 4)
    frac_captada    numeric,     -- fração da unidade dominante no filtro
    conf_geo3d      boolean      -- confiabilidade do cruzamento 3D
);

-- Matriz de treino materializada (Etapa 6) — colunas de covariáveis
-- adicionadas dinamicamente pelo estágio s05/s06 via ALTER TABLE.
CREATE TABLE matriz_treino (
    codigo_siagas text PRIMARY KEY REFERENCES pocos,
    alvo_log_qs   numeric,
    classe        text,          -- baixa | media | alta | muito_alta
    fold_espacial int            -- bloco da CV espacial
);
