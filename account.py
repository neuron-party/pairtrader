class Account:
    def __init__(self, cash_balance):
        self.cash_balance = cash_balance
        self.positions = dict()
    
    @property
    def asset_balance(self):
        asset_balance = 0
        if len(self.positions.values()) != 0:
            for n, p in self.positions.values():
                asset_balance += n*p
        return asset_balance
    
    @property
    def total_balance(self):
        return self.cash_balance + self.asset_balance
    
    def update_position(self, id, n, p):
        if n == 'close':
            n = -self.positions[id][0]
        self.cash_balance -= n*p
        if id not in self.positions:
            self.positions[id] = [n, p]
        else:
            n0, p0 = self.positions[id]
            self.positions[id][1] = p
            self.positions[id][0] += n
        
        # Remove empty positions
        if self.positions[id][0] == 0: 
            self.positions.pop(id)
        
