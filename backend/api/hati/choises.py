ANIMAL_CHOICES = [
    ('dog', 'Собака'),
]
ANIMAL_CHOICES = sorted(ANIMAL_CHOICES, key=lambda tup: tup[1])
ANIMAL_CHOICES.insert(0, ("undefined", "Не важно"))


TAIL_CHOICES = (
    ('long_tail', 'Длинный'),
    ('short_tail', 'Короткий/Нет хвоста'),
)
TAIL_CHOICES = sorted(TAIL_CHOICES, key=lambda tup: tup[1])
TAIL_CHOICES.insert(0, ("undefined", "Не важно"))


COLOR_CHOICES = (
    ('bright', 'Светлая'),
    ('dark', 'Тёмная'),
    ('multicolor', 'Многоцветная'),
)
COLOR_CHOICES = sorted(COLOR_CHOICES, key=lambda tup: tup[1])
COLOR_CHOICES.insert(0, ("undefined", "Не важно"))


BREAD_CHOICES = (
    ('kelpie', 'Австралийский келпи'),
    ('Australian_terrier', 'Австралийский терьер'),
    ('Eskimo_dog', 'Американская эскимосская собака'),
    ('American_Staffordshire_terrier', 'Американский стаффордширский терьер'),
    ('setter', 'Английский сеттер'),
    ('English_springer', 'Английский спрингер-спаниель'),
    ('Appenzeller', 'Аппенцеллер зенненхунд'),
    ('Afghan_hound', 'Афганская борзая'),
    ('affenpinscher', 'Аффенпинчер'),
    ('basenji', 'Басенджи'),
    ('basset', 'Бассет'),
    ('malinois', 'Бельгийская овчарка'),
    ('Bernese_mountain_dog', 'Бернский зенненхунд'),
    ('beagle', 'Бигль'),
    ('Blenheim_spaniel', 'Бленхейм спаниель'),
    ('Old_English_sheepdog', 'Бобтейл'),
    ('Greater_Swiss_Mountain_dog', 'Большой швейцарский зенненхунд'),
    ('Border_terrier', 'Бордер-терьер'),
    ('Boston_bull', 'Бостон-терьер'),
    ('Brittany_spaniel', 'Бретонский эпаньоль'),
    ('bull_mastiff', 'Бульмастиф'),
    ('Weimaraner', 'Веймаранер'),
    ('Pembroke', 'Вельш-коргипемброк'),
    ('vizsla', 'Венгерская выжла'),
    ('groenendael', 'Грюнендаль'),
    ('Scottish_deerhound', 'Дирхаунд'),
    ('Doberman', 'Доберман'),
    ('retriever', 'Золотистый ретривер'),
    ('Irish_wolfhound', 'Ирландский волкодав'),
    ('setter', 'Ирландский красный сеттер'),
    ('soft', 'Ирландский мягкошёрстный пшеничный терьер'),
    ('Irish_terrier', 'Ирландский терьер'),
    ('bloodhound', 'Ищейка'),
    ('Yorkshire_terrier', 'Йоркширский терьер'),
    ('keeshond', 'Кеесхонд'),
    ('clumber', 'Кламбер-спаниель'),
    ('komondor', 'Комондор'),
    ('redbone', 'Красный кунхаунд'),
    ('Mexican_hairless', 'Ксолоитцкуинтли'),
    ('German_short', 'Курцхаар'),
    ('curly', 'Курчавошёрстный ретривер'),
    ('retriever', 'Ретривер'),
    ('Labrador_retriever', 'Лабрадор-ретривер'),
    ('Italian_greyhound', 'Левретка'),
    ('Lakeland_terrier', 'Лейкленд-терьер'),
    ('Leonberg', 'Леонбергер'),
    ('Maltese_dog', 'Мальтийскаясобака'),
    ('poodle', 'Миниатюрный пудель'),
    ('standard_schnauzer', 'Миттельшнауцер'),
    ('pug', 'Мопс'),
    ('German_shepherd', 'Немецкая овчарка'),
    ('boxer', 'Немецкий боксёр'),
    ('Great_Dane', 'Немецкий дог'),
    ('Norwegian_elkhound', 'Норвежский серый элкхунд'),
    ('Newfoundland', 'Ньюфаундленд'),
    ('otterhound', 'Оттерхаунд'),
    ('papillon', 'Папийон'),
    ('Pekinese', 'Пекинес'),
    ('Great_Pyrenees', 'Пиренейская горная собака'),
    ('Ibizan_hound', 'Поденко ибиценко'),
    ('Pomeranian', 'Померанский шпиц'),
    ('retriever', 'Прямошёрстный ретривер'),
    ('Brabancon_griffon', 'Пти брабансон'),
    ('Rhodesian_ridgeback', 'Родезийский риджбек'),
    ('Rottweiler', 'Ротвейлер'),
    ('borzoi', 'Русская псовая борзая'),
    ('Saluki', 'Салюки'),
    ('Samoyed', 'Самоед'),
    ('Saint_Bernard', 'Сенбернар'),
    ('poodle', 'Стандартный пудель'),
    ('Staffordshire_bullterrier', 'Стаффордширский бультерьер'),
    ('Sussex_spaniel', 'Суссекс-спаниель'),
    ('schipperke', 'Схипперке'),
    ('Tibetan_mastiff', 'Тибетский мастиф'),
    ('Tibetan_terrier', 'Тибетский терьер'),
    ('poodle', 'Той-пудель'),
    ('toy_terrier', 'Тойтерьер'),
    ('whippet', 'Уиппет'),
    ('Bouvier_des_Flandres', 'Фландрский бувье'),
    ('chow', 'Чау-чау'),
    ('retriever', 'Чесапик-бей-ретривер'),
    ('Walker_hound', 'Чёрно-подпалый кунхаунд'),
    ('black', 'Чёрный терьер'),
    ('Shetland_sheepdog', 'Шелти'),
    ('setter', 'Шотландский сеттер'),
    ('EntleBucher', 'Энтлебухер зенненхунд'),
    ('Airedale', 'Эрдельтерьер'),
)
BREAD_CHOICES = sorted(BREAD_CHOICES, key=lambda tup: tup[1])
BREAD_CHOICES.insert(0, ("undefined", "Не важно"))
