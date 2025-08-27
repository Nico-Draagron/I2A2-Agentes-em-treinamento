# Cálculos de VR e dias úteis
"""
Módulo calculator.py
Responsável por cálculos financeiros precisos e detalhados do sistema VR.

Autor: Agente VR
Data: 2025-08
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import calendar
from core.rules import ResultadoElegibilidade, StatusElegibilidade

@dataclass
class CalculoVRDetalhado:
    """Resultado detalhado do cálculo de VR para um colaborador"""
    matricula: int
    nome_colaborador: Optional[str]
    data_admissao: Optional[datetime]
    sindicato: str
    estado: str
    competencia: str
    
    # Dados do período
    data_inicio_periodo: datetime
    data_fim_periodo: datetime
    dias_calendario_periodo: int
    dias_uteis_periodo: int
    dias_uteis_sindicato: int
    
    # Ajustes no período
    dias_ferias: int
    dias_afastamento: int
    dias_admissao_proporcional: int
    dias_desligamento_proporcional: int
    
    # Cálculo final
    dias_vr_pagos: int
    valor_diario_vr: Decimal
    valor_total_vr: Decimal
    valor_custo_empresa: Decimal  # 80%
    valor_desconto_colaborador: Decimal  # 20%
    
    # Detalhes contábeis
    centro_custo: Optional[str]
    conta_contabil_empresa: Optional[str]
    conta_contabil_colaborador: Optional[str]
    
    # Observações e auditoria
    observacoes: List[str]
    historico_calculo: Dict[str, Any]
    data_calculo: datetime
    
    # Validações
    calculo_validado: bool
    inconsistencias_encontradas: List[str]

@dataclass
class ResumoFinanceiroVR:
    """Resumo financeiro consolidado do VR"""
    competencia: str
    data_calculo: datetime
    
    # Totais gerais
    total_colaboradores: int
    total_dias_pagos: int
    valor_total_geral: Decimal
    valor_total_empresa: Decimal
    valor_total_colaboradores: Decimal
    
    # Por estado/sindicato
    resumo_por_estado: Dict[str, Dict[str, Any]]
    resumo_por_sindicato: Dict[str, Dict[str, Any]]
    
    # Distribuições
    distribuicao_por_faixa_valor: Dict[str, int]
    distribuicao_por_dias: Dict[int, int]
    
    # Métricas
    valor_medio_por_colaborador: Decimal
    valor_mediano_por_colaborador: Decimal
    dias_medio_por_colaborador: float

class VRCalculator:
    """
    Calculadora avançada de Vale Refeição com precisão contábil.
    """
    
    def __init__(self, competencia: str = "2025-05"):
        """
        Inicializa a calculadora de VR.
        
        Args:
            competencia: Competência no formato YYYY-MM
        """
        self.logger = logging.getLogger(__name__)
        self.competencia = competencia
        self.data_competencia = datetime.strptime(f"{competencia}-01", "%Y-%m-%d")
        
        # Calcular período da competência (dia 15 do mês anterior ao dia 15 do mês atual)
        if self.data_competencia.month == 1:
            mes_anterior = 12
            ano_anterior = self.data_competencia.year - 1
        else:
            mes_anterior = self.data_competencia.month - 1
            ano_anterior = self.data_competencia.year
        
        self.data_inicio_periodo = datetime(ano_anterior, mes_anterior, 15)
        self.data_fim_periodo = datetime(self.data_competencia.year, self.data_competencia.month, 15)
        
        # Configurações contábeis
        self.percentual_empresa = Decimal('0.80')
        self.percentual_colaborador = Decimal('0.20')
        self.precisao_decimal = 2
        
        # Feriados nacionais e regionais para 2025
        self.feriados_2025 = self._carregar_feriados_2025()
        
        # Contas contábeis padrão
        self.contas_contabeis = {
            'empresa': '2.1.01.003',  # Provisão VR - Empresa
            'colaborador': '1.2.01.015'  # VR a Descontar - Colaboradores
        }
        
        # Estatísticas de processamento
        self.estatisticas = {
            'total_calculados': 0,
            'valor_total_calculado': Decimal('0'),
            'tempo_processamento': None,
            'inconsistencias_calculo': 0
        }
    
    def calcular_vr_completo(
        self, 
        colaboradores_elegiveis: List[ResultadoElegibilidade],
        dados_adicionais: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Tuple[List[CalculoVRDetalhado], ResumoFinanceiroVR]:
        """
        Calcula VR completo para lista de colaboradores elegíveis.
        
        Args:
            colaboradores_elegiveis: Lista de colaboradores elegíveis do business_rules
            dados_adicionais: Dados adicionais opcionais (nome, centro de custo, etc.)
            
        Returns:
            Tuple com (lista_calculos_detalhados, resumo_financeiro)
        """
        self.logger.info("🧮 Iniciando cálculos detalhados de VR...")
        
        inicio_processamento = datetime.now()
        calculos_detalhados = []
        
        try:
            # Preparar dados auxiliares
            dados_auxiliares = self._preparar_dados_auxiliares(dados_adicionais)
            
            # Calcular VR para cada colaborador
            self.logger.info(f"   Calculando VR para {len(colaboradores_elegiveis)} colaboradores...")
            
            for colaborador in colaboradores_elegiveis:
                calculo = self._calcular_vr_individual(colaborador, dados_auxiliares)
                calculos_detalhados.append(calculo)
                
                # Atualizar estatísticas
                self._atualizar_estatisticas_calculo(calculo)
            
            # Gerar resumo financeiro
            resumo_financeiro = self._gerar_resumo_financeiro(calculos_detalhados)
            
            # Finalizar estatísticas
            fim_processamento = datetime.now()
            self.estatisticas['tempo_processamento'] = fim_processamento - inicio_processamento
            
            # Log do resumo
            self._log_resumo_calculos(calculos_detalhados, resumo_financeiro)
            
            return calculos_detalhados, resumo_financeiro
            
        except Exception as e:
            self.logger.error(f"❌ Erro nos cálculos de VR: {e}")
            raise
    
    def _calcular_vr_individual(
        self, 
        colaborador: ResultadoElegibilidade,
        dados_auxiliares: Dict[str, Any]
    ) -> CalculoVRDetalhado:
        """Calcula VR detalhado para um colaborador individual"""
        
        matricula = colaborador.matricula
        observacoes = []
        historico_calculo = {}
        inconsistencias = []
        
        # Obter dados auxiliares do colaborador
        dados_colaborador = dados_auxiliares.get(matricula, {})
        
        # 1. Calcular dias do período com precisão
        dias_periodo = self._calcular_dias_periodo_preciso(
            colaborador.data_inicio_periodo,
            colaborador.data_fim_periodo,
            colaborador.estado_sindicato
        )
        
        historico_calculo['dias_periodo'] = dias_periodo
        
        # 2. Ajustar dias considerando todas as variáveis
        dias_ajustados = self._calcular_dias_ajustados(colaborador, dados_colaborador)
        
        historico_calculo['dias_ajustados'] = dias_ajustados
        observacoes.extend(dias_ajustados.get('observacoes', []))
        
        # 3. Calcular valores financeiros com precisão decimal
        valores = self._calcular_valores_financeiros(
            dias_ajustados['dias_finais'],
            colaborador.valor_diario
        )
        
        historico_calculo['valores'] = valores
        
        # 4. Validações finais
        validacoes = self._validar_calculo_individual(colaborador, dias_ajustados, valores)
        inconsistencias.extend(validacoes['inconsistencias'])
        
        # 5. Obter informações contábeis
        info_contabil = self._obter_informacoes_contabeis(colaborador, dados_colaborador)
        
        # 6. Montar resultado final
        calculo_detalhado = CalculoVRDetalhado(
            matricula=matricula,
            nome_colaborador=dados_colaborador.get('nome'),
            data_admissao=dados_colaborador.get('data_admissao'),
            sindicato=colaborador.estado_sindicato,  # Usar estado como sindicato resumido
            estado=colaborador.estado_sindicato,
            competencia=self.competencia,
            
            # Período
            data_inicio_periodo=colaborador.data_inicio_periodo,
            data_fim_periodo=colaborador.data_fim_periodo,
            dias_calendario_periodo=dias_periodo['dias_calendario'],
            dias_uteis_periodo=dias_periodo['dias_uteis'],
            dias_uteis_sindicato=colaborador.dias_uteis_sindicato,
            
            # Ajustes
            dias_ferias=colaborador.dias_ferias,
            dias_afastamento=dias_ajustados.get('dias_afastamento', 0),
            dias_admissao_proporcional=dias_ajustados.get('dias_admissao_prop', 0),
            dias_desligamento_proporcional=dias_ajustados.get('dias_desligamento_prop', 0),
            
            # Cálculo final
            dias_vr_pagos=dias_ajustados['dias_finais'],
            valor_diario_vr=self._to_decimal(colaborador.valor_diario),
            valor_total_vr=valores['valor_total'],
            valor_custo_empresa=valores['valor_empresa'],
            valor_desconto_colaborador=valores['valor_colaborador'],
            
            # Contábil
            centro_custo=info_contabil.get('centro_custo'),
            conta_contabil_empresa=self.contas_contabeis['empresa'],
            conta_contabil_colaborador=self.contas_contabeis['colaborador'],
            
            # Auditoria
            observacoes=observacoes,
            historico_calculo=historico_calculo,
            data_calculo=datetime.now(),
            
            # Validações
            calculo_validado=len(inconsistencias) == 0,
            inconsistencias_encontradas=inconsistencias
        )
        
        return calculo_detalhado
    
    def _calcular_dias_periodo_preciso(
        self, 
        data_inicio: datetime, 
        data_fim: datetime, 
        estado: str
    ) -> Dict[str, int]:
        """Calcula dias do período considerando feriados regionais"""
        
        dias_calendario = (data_fim - data_inicio).days
        
        # Calcular dias úteis considerando feriados
        dias_uteis = 0
        data_atual = data_inicio
        
        while data_atual < data_fim:
            # Segunda a sexta (0-4)
            if data_atual.weekday() < 5:
                # Verificar se não é feriado
                if not self._eh_feriado(data_atual, estado):
                    dias_uteis += 1
            
            data_atual += timedelta(days=1)
        
        return {
            'dias_calendario': dias_calendario,
            'dias_uteis': dias_uteis
        }
    
    def _calcular_dias_ajustados(
        self, 
        colaborador: ResultadoElegibilidade,
        dados_colaborador: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcula dias ajustados considerando todas as variáveis"""
        
        observacoes = []
        
        # Começar com dias elegíveis do business_rules
        dias_base = colaborador.dias_elegivel
        
        # Ajustes específicos
        dias_afastamento = 0
        dias_admissao_prop = 0
        dias_desligamento_prop = 0
        
        # Se é proporcional, pode ter ajustes adicionais
        if colaborador.status == StatusElegibilidade.PROPORCIONAL:
            # Verificar se há dados de admissão no período
            if dados_colaborador.get('data_admissao'):
                data_admissao = dados_colaborador['data_admissao']
                if data_admissao > colaborador.data_inicio_periodo:
                    dias_perdidos_admissao = (data_admissao - colaborador.data_inicio_periodo).days
                    dias_admissao_prop = min(dias_perdidos_admissao, dias_base)
                    observacoes.append(f"Ajuste admissão: -{dias_admissao_prop} dias")
        
        # Dias finais não podem ser negativos
        dias_finais = max(0, dias_base)
        
        return {
            'dias_finais': dias_finais,
            'dias_afastamento': dias_afastamento,
            'dias_admissao_prop': dias_admissao_prop,
            'dias_desligamento_prop': dias_desligamento_prop,
            'observacoes': observacoes
        }
    
    def _calcular_valores_financeiros(self, dias_vr: int, valor_diario: float) -> Dict[str, Decimal]:
        """Calcula valores financeiros com precisão decimal"""
        
        # Converter para Decimal para precisão
        valor_diario_decimal = self._to_decimal(valor_diario)
        dias_decimal = Decimal(str(dias_vr))
        
        # Cálculo base
        valor_total = dias_decimal * valor_diario_decimal
        
        # Divisão empresa/colaborador
        valor_empresa = valor_total * self.percentual_empresa
        valor_colaborador = valor_total * self.percentual_colaborador
        
        # Arredondar para centavos
        valor_total = self._arredondar_moeda(valor_total)
        valor_empresa = self._arredondar_moeda(valor_empresa)
        valor_colaborador = self._arredondar_moeda(valor_colaborador)
        
        # Garantir que empresa + colaborador = total (ajustar diferenças de arredondamento)
        soma_parcelas = valor_empresa + valor_colaborador
        if soma_parcelas != valor_total:
            diferenca = valor_total - soma_parcelas
            valor_empresa += diferenca  # Ajustar na empresa
        
        return {
            'valor_total': valor_total,
            'valor_empresa': valor_empresa,
            'valor_colaborador': valor_colaborador
        }
    
    def _validar_calculo_individual(
        self, 
        colaborador: ResultadoElegibilidade,
        dias_ajustados: Dict[str, Any],
        valores: Dict[str, Decimal]
    ) -> Dict[str, List[str]]:
        """Valida cálculo individual e retorna inconsistências"""
        
        inconsistencias = []
        
        # Validação 1: Dias não podem ser negativos
        if dias_ajustados['dias_finais'] < 0:
            inconsistencias.append("Dias finais negativos")
        
        # Validação 2: Dias não podem exceder limite do sindicato
        if dias_ajustados['dias_finais'] > colaborador.dias_uteis_sindicato:
            inconsistencias.append(f"Dias finais ({dias_ajustados['dias_finais']}) excedem limite do sindicato ({colaborador.dias_uteis_sindicato})")
        
        # Validação 3: Valor não pode ser zero se há dias
        if dias_ajustados['dias_finais'] > 0 and valores['valor_total'] == 0:
            inconsistencias.append("Valor total zero com dias positivos")
        
        # Validação 4: Proporção empresa + colaborador deve ser = total
        soma_parcelas = valores['valor_empresa'] + valores['valor_colaborador']
        if abs(soma_parcelas - valores['valor_total']) > Decimal('0.01'):
            inconsistencias.append(f"Inconsistência na divisão: {soma_parcelas} ≠ {valores['valor_total']}")
        
        # Validação 5: Valor diário deve ser positivo
        if colaborador.valor_diario <= 0:
            inconsistencias.append("Valor diário inválido")
        
        return {'inconsistencias': inconsistencias}
    
    def _gerar_resumo_financeiro(self, calculos: List[CalculoVRDetalhado]) -> ResumoFinanceiroVR:
        """Gera resumo financeiro consolidado"""
        
        if not calculos:
            return self._resumo_vazio()
        
        # Totais gerais
        total_colaboradores = len(calculos)
        total_dias_pagos = sum(c.dias_vr_pagos for c in calculos)
        valor_total_geral = sum(c.valor_total_vr for c in calculos)
        valor_total_empresa = sum(c.valor_custo_empresa for c in calculos)
        valor_total_colaboradores = sum(c.valor_desconto_colaborador for c in calculos)
        
        # Resumos por agrupamento
        resumo_por_estado = self._agrupar_por_estado(calculos)
        resumo_por_sindicato = self._agrupar_por_sindicato(calculos)
        
        # Distribuições
        distribuicao_faixa_valor = self._calcular_distribuicao_valores(calculos)
        distribuicao_dias = self._calcular_distribuicao_dias(calculos)
        
        # Métricas
        valores = [c.valor_total_vr for c in calculos]
        valor_medio = valor_total_geral / Decimal(str(total_colaboradores))
        valor_mediano = self._calcular_mediana_decimal(valores)
        dias_medio = total_dias_pagos / total_colaboradores
        
        return ResumoFinanceiroVR(
            competencia=self.competencia,
            data_calculo=datetime.now(),
            total_colaboradores=total_colaboradores,
            total_dias_pagos=total_dias_pagos,
            valor_total_geral=valor_total_geral,
            valor_total_empresa=valor_total_empresa,
            valor_total_colaboradores=valor_total_colaboradores,
            resumo_por_estado=resumo_por_estado,
            resumo_por_sindicato=resumo_por_sindicato,
            distribuicao_por_faixa_valor=distribuicao_faixa_valor,
            distribuicao_por_dias=distribuicao_dias,
            valor_medio_por_colaborador=self._arredondar_moeda(valor_medio),
            valor_mediano_por_colaborador=valor_mediano,
            dias_medio_por_colaborador=dias_medio
        )
    
    def _carregar_feriados_2025(self) -> Dict[str, List[datetime]]:
        """Carrega feriados nacionais e regionais para 2025"""
        return {
            'nacionais': [
                datetime(2025, 1, 1),   # Confraternização Universal
                datetime(2025, 4, 18),  # Sexta-feira Santa
                datetime(2025, 4, 21),  # Tiradentes
                datetime(2025, 5, 1),   # Dia do Trabalhador
                datetime(2025, 9, 7),   # Independência do Brasil
                datetime(2025, 10, 12), # Nossa Senhora Aparecida
                datetime(2025, 11, 2),  # Finados
                datetime(2025, 11, 15), # Proclamação da República
                datetime(2025, 12, 25), # Natal
            ],
            'São Paulo': [
                datetime(2025, 2, 13),  # Carnaval (facultativo)
                datetime(2025, 6, 19),  # Corpus Christi (facultativo)
            ],
            'Rio de Janeiro': [
                datetime(2025, 4, 23),  # São Jorge
                datetime(2025, 11, 20), # Zumbi dos Palmares
            ],
            'Rio Grande do Sul': [
                datetime(2025, 9, 20),  # Revolução Farroupilha
            ],
            'Paraná': [
                datetime(2025, 12, 19), # Emancipação do Paraná
            ]
        }
    
    def _eh_feriado(self, data: datetime, estado: str) -> bool:
        """Verifica se uma data é feriado nacional ou regional"""
        # Feriados nacionais
        if data in self.feriados_2025['nacionais']:
            return True
        
        # Feriados regionais
        if estado in self.feriados_2025 and data in self.feriados_2025[estado]:
            return True
        
        return False
    
    def _preparar_dados_auxiliares(self, dados_adicionais: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Any]]:
        """Prepara dados auxiliares dos colaboradores"""
        dados_aux = {}
        
        if dados_adicionais:
            # Processar dados de admissão
            if 'admissoes' in dados_adicionais:
                for _, row in dados_adicionais['admissoes'].iterrows():
                    matricula = int(row['matricula'])
                    dados_aux[matricula] = {
                        'data_admissao': row.get('data_admissao'),
                        'nome': row.get('nome'),
                        'centro_custo': row.get('centro_custo')
                    }
            
            # Processar outros dados auxiliares
            if 'ativos' in dados_adicionais:
                for _, row in dados_adicionais['ativos'].iterrows():
                    matricula = int(row['matricula'])
                    if matricula not in dados_aux:
                        dados_aux[matricula] = {}
                    dados_aux[matricula].update({
                        'nome': row.get('nome'),
                        'cargo': row.get('cargo'),
                        'centro_custo': row.get('centro_custo')
                    })
        
        return dados_aux
    
    def _obter_informacoes_contabeis(self, colaborador: ResultadoElegibilidade, dados_colaborador: Dict[str, Any]) -> Dict[str, str]:
        """Obtém informações contábeis do colaborador"""
        return {
            'centro_custo': dados_colaborador.get('centro_custo', '999999'),
            'conta_empresa': self.contas_contabeis['empresa'],
            'conta_colaborador': self.contas_contabeis['colaborador']
        }
    
    def _agrupar_por_estado(self, calculos: List[CalculoVRDetalhado]) -> Dict[str, Dict[str, Any]]:
        """Agrupa cálculos por estado"""
        resumo = {}
        
        for calculo in calculos:
            estado = calculo.estado
            if estado not in resumo:
                resumo[estado] = {
                    'colaboradores': 0,
                    'dias_totais': 0,
                    'valor_total': Decimal('0'),
                    'valor_empresa': Decimal('0'),
                    'valor_colaboradores': Decimal('0')
                }
            
            resumo[estado]['colaboradores'] += 1
            resumo[estado]['dias_totais'] += calculo.dias_vr_pagos
            resumo[estado]['valor_total'] += calculo.valor_total_vr
            resumo[estado]['valor_empresa'] += calculo.valor_custo_empresa
            resumo[estado]['valor_colaboradores'] += calculo.valor_desconto_colaborador
        
        return resumo
    
    def _agrupar_por_sindicato(self, calculos: List[CalculoVRDetalhado]) -> Dict[str, Dict[str, Any]]:
        """Agrupa cálculos por sindicato"""
        return self._agrupar_por_estado(calculos)  # Mesmo agrupamento por enquanto
    
    def _calcular_distribuicao_valores(self, calculos: List[CalculoVRDetalhado]) -> Dict[str, int]:
        """Calcula distribuição por faixas de valor"""
        faixas = {
            'R$ 0-200': 0,
            'R$ 201-400': 0,
            'R$ 401-600': 0,
            'R$ 601-800': 0,
            'R$ 801+': 0
        }
        
        for calculo in calculos:
            valor = float(calculo.valor_total_vr)
            
            if valor <= 200:
                faixas['R$ 0-200'] += 1
            elif valor <= 400:
                faixas['R$ 201-400'] += 1
            elif valor <= 600:
                faixas['R$ 401-600'] += 1
            elif valor <= 800:
                faixas['R$ 601-800'] += 1
            else:
                faixas['R$ 801+'] += 1
        
        return faixas
    
    def _calcular_distribuicao_dias(self, calculos: List[CalculoVRDetalhado]) -> Dict[int, int]:
        """Calcula distribuição por dias de VR"""
        distribuicao = {}
        
        for calculo in calculos:
            dias = calculo.dias_vr_pagos
            distribuicao[dias] = distribuicao.get(dias, 0) + 1
        
        return distribuicao
    
    def _to_decimal(self, valor: float) -> Decimal:
        """Converte float para Decimal com precisão"""
        return Decimal(str(valor)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def _arredondar_moeda(self, valor: Decimal) -> Decimal:
        """Arredonda valor para centavos"""
        return valor.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def _calcular_mediana_decimal(self, valores: List[Decimal]) -> Decimal:
        """Calcula mediana de valores Decimal"""
        valores_ordenados = sorted(valores)
        n = len(valores_ordenados)
        
        if n % 2 == 0:
            mediana = (valores_ordenados[n//2 - 1] + valores_ordenados[n//2]) / Decimal('2')
        else:
            mediana = valores_ordenados[n//2]
        
        return self._arredondar_moeda(mediana)
    
    def _resumo_vazio(self) -> ResumoFinanceiroVR:
        """Retorna resumo vazio para casos sem dados"""
        return ResumoFinanceiroVR(
            competencia=self.competencia,
            data_calculo=datetime.now(),
            total_colaboradores=0,
            total_dias_pagos=0,
            valor_total_geral=Decimal('0'),
            valor_total_empresa=Decimal('0'),
            valor_total_colaboradores=Decimal('0'),
            resumo_por_estado={},
            resumo_por_sindicato={},
            distribuicao_por_faixa_valor={},
            distribuicao_por_dias={},
            valor_medio_por_colaborador=Decimal('0'),
            valor_mediano_por_colaborador=Decimal('0'),
            dias_medio_por_colaborador=0.0
        )
    
    def _atualizar_estatisticas_calculo(self, calculo: CalculoVRDetalhado) -> None:
        """Atualiza estatísticas do processamento"""
        self.estatisticas['total_calculados'] += 1
        self.estatisticas['valor_total_calculado'] += calculo.valor_total_vr
        
        if not calculo.calculo_validado:
            self.estatisticas['inconsistencias_calculo'] += 1
    
    def _log_resumo_calculos(self, calculos: List[CalculoVRDetalhado], resumo: ResumoFinanceiroVR) -> None:
        """Registra resumo dos cálculos realizados"""
        self.logger.info("📊 RESUMO DOS CÁLCULOS DE VR:")
        self.logger.info(f"   Colaboradores calculados: {len(calculos)}")
        self.logger.info(f"   Total de dias pagos: {resumo.total_dias_pagos}")
        self.logger.info(f"   Valor total: R$ {resumo.valor_total_geral:,.2f}")
        self.logger.info(f"   Custo empresa: R$ {resumo.valor_total_empresa:,.2f}")
        self.logger.info(f"   Desconto colaboradores: R$ {resumo.valor_total_colaboradores:,.2f}")
        self.logger.info(f"   Valor médio por colaborador: R$ {resumo.valor_medio_por_colaborador:,.2f}")
        
        # Inconsistências
        inconsistencias_total = sum(1 for c in calculos if not c.calculo_validado)
        if inconsistencias_total > 0:
            self.logger.warning(f"   ⚠️ Cálculos com inconsistências: {inconsistencias_total}")
    
    def gerar_relatorio_calculos(self, calculos: List[CalculoVRDetalhado]) -> pd.DataFrame:
        """
        Gera relatório detalhado dos cálculos realizados.
        
        Args:
            calculos: Lista de cálculos detalhados
            
        Returns:
            DataFrame com relatório dos cálculos
        """
        dados_relatorio = []
        
        for calculo in calculos:
            dados_relatorio.append({
                'Matrícula': calculo.matricula,
                'Nome': calculo.nome_colaborador or '',
                'Estado': calculo.estado,
                'Competência': calculo.competencia,
                'Dias VR': calculo.dias_vr_pagos,
                'Dias Úteis Sindicato': calculo.dias_uteis_sindicato,
                'Dias Férias': calculo.dias_ferias,
                'Valor Diário': float(calculo.valor_diario_vr),
                'Valor Total': float(calculo.valor_total_vr),
                'Custo Empresa': float(calculo.valor_custo_empresa),
                'Desconto Colaborador': float(calculo.valor_desconto_colaborador),
                'Centro Custo': calculo.centro_custo or '',
                'Validado': calculo.calculo_validado,
                'Inconsistências': '; '.join(calculo.inconsistencias_encontradas),
                'Observações': '; '.join(calculo.observacoes),
                'Data Cálculo': calculo.data_calculo.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return pd.DataFrame(dados_relatorio)
    
    def salvar_relatorio_calculos(self, calculos: List[CalculoVRDetalhado], caminho: str) -> None:
        """
        Salva relatório de cálculos em Excel.
        
        Args:
            calculos: Lista de cálculos
            caminho: Caminho do arquivo de saída
        """
        relatorio = self.gerar_relatorio_calculos(calculos)
        relatorio.to_excel(caminho, index=False)
        self.logger.info(f"📊 Relatório de cálculos salvo em: {caminho}")


# Função utilitária para uso direto
def calcular_vr_colaboradores(
    colaboradores_elegiveis: List[ResultadoElegibilidade],
    competencia: str = "2025-05",
    dados_adicionais: Optional[Dict[str, pd.DataFrame]] = None
) -> Tuple[List[CalculoVRDetalhado], ResumoFinanceiroVR]:
    """
    Função utilitária para calcular VR de colaboradores elegíveis.
    
    Args:
        colaboradores_elegiveis: Lista de colaboradores elegíveis do business_rules
        competencia: Competência no formato YYYY-MM
        dados_adicionais: Dados adicionais opcionais
        
    Returns:
        Tuple com (calculos_detalhados, resumo_financeiro)
    """
    calculator = VRCalculator(competencia)
    return calculator.calcular_vr_completo(colaboradores_elegiveis, dados_adicionais)


# Exemplo de uso
if __name__ == "__main__":
    import logging
    from data_reader import carregar_bases_vr
    from data_validator import validar_bases_vr
    from business_rules import processar_regras_negocio_vr
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Testar cálculos
    try:
        # Pipeline completo até cálculos
        bases = carregar_bases_vr("data/input")
        bases_validadas, _ = validar_bases_vr(bases)
        resultados_elegibilidade = processar_regras_negocio_vr(bases_validadas)
        
        # Filtrar apenas elegíveis
        from business_rules import BusinessRulesEngine
        engine = BusinessRulesEngine()
        colaboradores_elegiveis = engine.filtrar_elegiveis(resultados_elegibilidade)
        
        # Calcular VR
        calculos, resumo = calcular_vr_colaboradores(colaboradores_elegiveis, dados_adicionais=bases_validadas)
        
        print("✅ Cálculos de VR concluídos!")
        print(f"📊 Colaboradores calculados: {len(calculos)}")
        print(f"💰 Valor total: R$ {resumo.valor_total_geral:,.2f}")
        print(f"🏢 Custo empresa: R$ {resumo.valor_total_empresa:,.2f}")
        print(f"👤 Desconto colaboradores: R$ {resumo.valor_total_colaboradores:,.2f}")
        
        # Mostrar exemplos
        print("\n📝 Exemplos de cálculos:")
        for calculo in calculos[:3]:
            print(f"   Matrícula {calculo.matricula}: {calculo.dias_vr_pagos} dias × R$ {calculo.valor_diario_vr} = R$ {calculo.valor_total_vr}")
        
        # Verificar distribuição por estado
        print(f"\n📊 Distribuição por estado:")
        for estado, dados in resumo.resumo_por_estado.items():
            print(f"   {estado}: {dados['colaboradores']} colaboradores, R$ {dados['valor_total']:,.2f}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")