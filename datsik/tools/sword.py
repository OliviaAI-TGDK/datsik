def offer_sword(self):
    """Major semblance mechanism — all flows must call this."""
    kata = f"[Sword Kata] {self.sword} Sword offered, kata sealed."
    self.kata.append(kata)        
    logging.info(kata)
    return kata