# Face ID

## Requirement
```bash
pip3 install face_recognition
```

## Experiments
- `raw/`: solo-pictures for known students
    + `1.jpg`: for student_1
    + `2.jpg`: for student_2
    + `3.jpg`: for student_3

### a.png
```bash
python3 main.py a.png
```
```text
Guess face_0 is student 1 with confidence score 0.392
Guess face_1 is student 1 with confidence score 0.519
```

### b.jpg
```bash
python3 main.py b.jpg
```
```text
Guess face_0 is student 2 with confidence score 0.422
Guess face_1 is student 3 with confidence score 0.409
Guess face_2 is student 1 with confidence score 0.557
```

### c.jpg
```bash
python3 main.py c.jpg
```
```text
Guess face_0 is student 1 with confidence score 0.417
Guess face_1 is student 3 with confidence score 0.522
```

