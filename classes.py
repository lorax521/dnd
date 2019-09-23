from random import seed
from random import random
seed(2)

def genRandom(min, max):
    scaledValue = min + (random() * (max - min))
    return scaledValue

class Base:
    def __init__(self):
        self.stats: dict = {
            'strength': {'base': 0, 'modifier': 0},
            'dexterity': {'base': 0, 'modifier': 0},
            'consitution': {'base': 0, 'modifier': 0},
            'intelligence': {'base': 0, 'modifier': 0},
            'wisdom': {'base': 0, 'modifier': 0},
            'charisma': {'base': 0, 'modifier': 0}
        }
        self.armorClass = 0
        self.hitPoints = 0
        self.initiative = 0 


class Monster(Base):
    def __init__(self, difficulty, level):
        super().__init__()
        Base.damage = self.rollDamage(difficulty)
        Base.armorClass = self.rollArmorClass(difficulty)
        Base.hitPoints = self.rollHitPoints(difficulty)
        Base.stats = self.rollStats(difficulty)
        Base.initative = self.rollInitiative(difficulty)

        # self.difficultyLevels = {
        #     'chump': 0,
        #     'easy': 1,
        #     'medium': 2,
        #     'hard': 3,
        #     'diablo': 4,
        # }

        # self.difficulty = difficulty
        # if type(difficulty) == str:
        #     try:
        #         difficulty = self.difficultyLevels[difficulty]
        #     except:
        #         print('Difficultly level not found. Assigning level medium by default.')
        #         difficulty = 2

    def rollDamage(self, difficulty):
        maxValue = difficulty * 3
        minValue = difficulty * 2
        self.damage = int(genRandom(minValue, maxValue) + 0.5)

    def rollArmorClass(self, difficulty):
        maxValue = difficulty * 6
        minValue = difficulty * 4
        self.armorClass = int(genRandom(minValue, maxValue) + 0.5)

    def rollHitPoints(self, difficulty):
        maxValue = difficulty * 6
        minValue = difficulty * 4
        self.hitPoints = int(genRandom(minValue, maxValue) + 0.5)

    def rollStats(self, difficulty):
        maxValue = difficulty * 6
        minValue = difficulty * 4
        maxMod = difficulty * 1
        minMod = difficulty - 1 if difficulty - 1 >= 0 else 0
        for stat in self.stats:
            self.stats[stat]['base'] = int(genRandom(minValue, maxValue) + 0.5)
            self.stats[stat]['modifier'] = int(genRandom(minMod, maxMod) + 0.5)
    
    def rollInitiative(self, difficulty):
        maxValue = difficulty * 6
        minValue = difficulty * 4
        self.initative = int(genRandom(minValue, maxValue) + 0.5)
        # return int(genRandom(minValue, maxValue) + 0.5)


class Character(Base):
    def __init__(self, charRace, charClass, level):
        super().__init__()
        # ranges for stats by race and class
        raceReference = {
            'elf': {
                'hitPoints': {'low': 10, 'high': 16},
                'stats': {'low': 8, 'high': 18}
            }
        }
        classReference = {
            'druid': {
                'damage': {'low': 4, 'high': 8},
                'armorClass': {'low': 12, 'high': 18},
            }
        }
        # reassign race and class based on reference dict
        self.charRace = raceReference[charRace]
        self.charClass = classReference[charClass]
        Base.damage = self.rollDamage()
        Base.armorClass = self.rollArmorClass()
        Base.hitPoints = self.rollHitPoints()
        Base.stats = self.rollStats()
        Base.initative = self.rollInitiative()
    

    def rollDamage(self):
        self.damage = int(genRandom(self.charClass['damage']['low'], self.charClass['damage']['high']) + 0.5)

    def rollArmorClass(self):
        self.armorClass = int(genRandom(self.charClass['armorClass']['low'], self.charClass['armorClass']['high']) + 0.5)

    def rollHitPoints(self):
        self.hitPoints = int(genRandom(self.charRace['hitPoints']['low'], self.charRace['hitPoints']['high']) + 0.5)

    def rollStats(self):
        for stat in self.stats:
            newStat = int(genRandom(self.charRace['stats']['low'], self.charRace['stats']['high']) + 0.5)
            newMod = int(genRandom(1, 2) + 0.5)
            self.stats[stat]['base'] = newStat
            self.stats[stat]['modifier'] = newMod

    def rollInitiative(self):
        self.initiative = int(genRandom(self.charRace['stats']['low'], self.charRace['stats']['high']) + 0.5)
