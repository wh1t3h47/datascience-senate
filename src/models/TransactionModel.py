from dataclasses import dataclass
from datetime import date, datetime
from json import JSONEncoder

@dataclass
class TransactionModel():
    name: str
    owner: str
    asset_description: str
    asset_type: str
    type: str
    amount: float
    comment: str
    created_at: date
    # Preço média do período de aplicação até o final (suavizada)
    # Exemplo 1y -> Preço médio de todos os 365 dias depois da
    # aplicação
    price_avarage_in_1d: int
    price_avarage_in_7d: int
    price_avarage_in_15d: int
    price_avarage_in_1m: int
    price_avarage_in_6m: int
    price_avarage_in_1y: int
    # Preço duma média diária a partir do prazo inicial
    # Exemplo: 7d -> Preço médio do dia 7 dias depois da data
    # de aplicação
    # Por enquanto independe do prazo de retirada
    # @todo avaliar uma estratégia de parear compras e vendas
    # e ver se melhora o modelo
    price_daily_after_1d: int
    price_daily_after_7d: int
    price_daily_after_15d: int
    price_daily_after_1m: int
    price_daily_after_6m: int
    price_daily_after_1y: int

class TransactionEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TransactionModel):
            # Convert date to string
            obj.created_at = obj.created_at.isoformat() if obj.created_at else None
            return obj.__dict__
        return super().default(obj)
