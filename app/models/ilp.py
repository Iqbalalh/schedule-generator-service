import json
import httpx
import os
from dotenv import load_dotenv
from pulp import LpMinimize, LpProblem, LpVariable, lpSum
from collections import defaultdict

load_dotenv()

# Preprocessing Data: Menyesuaikan dari style JS ke style Python
def preprocess_data(data):
    if isinstance(data, dict):
        return {key: preprocess_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [preprocess_data(item) for item in data]
    elif data is True:
        return True
    elif data is False:
        return False
    elif data is None:
        return None
    else:
        return data

# Fetch data yang diperlukan
def fetch_data(params):
    url = os.getenv("MAIN_SERVICE_GET_DATA")
    try:
        response = httpx.get(url, params=params)
        response.raise_for_status()
        raw_data = response.json()
        return preprocess_data(raw_data)  # Preprocess data
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting data: {exc}")
        return {}
    except httpx.HTTPStatusError as exc:
        print(f"HTTP error occurred: {exc}")
        return {}

# Fungsi utama penjadwalan
def schedule_classes(data):
    rooms = data["rooms"]
    lecturers = data["lecturers"]
    classLecturers = data["classLecturers"]
    days = data["scheduleDays"]
    sessions = data["scheduleSessions"]

    # Mendefinisikan model ILP
    model = LpProblem("Classroom_Scheduling", LpMinimize)

    # Mendeklarasikan Decision Variables
    x = {
        (classLecturer["id"], room["id"], day["id"], session["id"]): LpVariable(
            f"x_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}", cat="Binary"
        )
        for classLecturer in classLecturers
        for room in rooms
        for day in days
        for session in sessions
    }

    y = {
        (lecturer["id"], day["id"], session["id"]): LpVariable(
            f"y_{lecturer['id']}_{day['id']}_{session['id']}", cat="Binary"
        )
        for lecturer in lecturers
        for day in days
        for session in sessions
    }

    # Mendeklarasikan Objective Function
    model += (
        lpSum(
            x[classLecturer["id"], room["id"], day["id"], session["id"]] * (room["roomCapacity"] - classLecturer["class"]["classCapacity"])
            for classLecturer in classLecturers
            for room in rooms
            for day in days
            for session in sessions
            if room["roomCapacity"] > classLecturer["class"]["classCapacity"]
        )
        + lpSum(
            y[lecturer["id"], day["id"], session["id"]] for lecturer in lecturers for day in days for session in sessions
        ),
        "Minimize_Objective",
    )

    # Constraints
                
    # 1. Tiap pertemuan dijadwalkan tepat satu dan tidak boleh lebih
    for classLecturer in classLecturers:
        model += (
            lpSum(x[classLecturer["id"], room["id"], day["id"], session["id"]] for room in rooms for day in days for session in sessions)
            == 1,
            f"classLecturer_{classLecturer['id']}_Scheduled_Once",
        )

    # 2. Tiap ruangan tidak boleh digunakan oleh lebih dari satu kelas pada waktu yang sama
    for room in rooms:
        for day in days:
            for session in sessions:
                model += (
                    lpSum(x[classLecturer["id"], room["id"], day["id"], session["id"]] for classLecturer in classLecturers) <= 1,
                    f"Room_{room['id']}_Single_Class_{day['id']}_{session['id']}",
                )

    # 3. Untuk menggunakan ruangan kelas, kapasitas ruangan harus lebih besar atau sama dengan kapasitas mahasiswa pada suatu kelas
    for classLecturer in classLecturers:
        subject_type_id = classLecturer["class"]["subSubject"]["subjectTypeId"]
        for room in rooms:
            if room["roomType"] == "Kelas":  # Only apply for rooms of type "Kelas"
                for day in days:
                    for session in sessions:
                        if subject_type_id in [1, 2]:  # Only for teori and responsi
                            model += (
                                x[classLecturer["id"], room["id"], day["id"], session["id"]] * classLecturer["class"]["classCapacity"]
                                <= room["roomCapacity"],
                                f"Room_Capacity_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}",
                            )
                            # Praktikum tidak harus mengikuti constraint ini

    # 4. Tiap dosen hanya boleh mengajar maksimal tiga sesi dalam sehari
    for lecturer in lecturers:
        for day in days:
            model += (
                lpSum(
                    x[classLecturer["id"], room["id"], day["id"], session["id"]]
                    for classLecturer in classLecturers
                    for room in rooms
                    for session in sessions
                    if (classLecturer["primaryLecturerId"] == lecturer["id"] or classLecturer["secondaryLecturerId"] == lecturer["id"])
                    and classLecturer["class"]["subSubject"]["subjectTypeId"] == 1  # Hanya untuk kelas teori
                ) <= 3,
                f"Lecturer_{lecturer['id']}_Max_3_Theory_Sessions_{day['id']}",
            )
    
    # 5. Pada hari Jumat, perkuliahan hanya akan dijadwalkan pada sesi 1, 2, 4, dan 5
    for day in days:
        if day["day"] == "Jumat": 
            for classLecturer in classLecturers:
                for room in rooms:
                    for session in sessions:
                        if session["id"] == 3:
                            model += (
                                x[classLecturer["id"], room["id"], day["id"], session["id"]] == 0,
                                f"No_3rd_Session_On_Friday_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}",
                            )
    
    class_groups = defaultdict(list)
    for classLecturer in classLecturers:
        class_name_id = classLecturer["class"]["studyProgramClassId"]
        semester_id = classLecturer["class"]["subSubject"]["subject"]["semesterId"]
        class_id = classLecturer["classId"]
        is_practicum = classLecturer["class"]["subSubject"]["subjectTypeId"] == 3
        class_groups[(class_name_id, semester_id, class_id, is_practicum)].append(classLecturer["id"])

    blocked_sets = [
        {1, 5, 7, 9},
        {2, 5, 7, 9},
        {3, 6, 7, 10},
        {4, 6, 7, 10},
    ]

    for (class_name_id, semester_id, class_id, is_practicum), classLecGroup in class_groups.items():
        for day in days:
            for session in sessions:
                if len(classLecGroup) > 1:
    # 6. Untuk kelas yang bukan praktikum serta memiliki nama dan semester yang sama hanya boleh dijadwalkan tepat satu kali pada hari dan sesi yang sama
                    if not is_practicum:
                        for blocked_set in blocked_sets:
                            for classLec1 in classLecGroup:
                                for classLec2 in classLecGroup:
                                    if classLec1 != classLec2 and classLec1 in blocked_set and classLec2 in blocked_set:
                                        model += (
                                            lpSum(
                                                x[classLec1, room["id"], day["id"], session["id"]]
                                                for room in rooms
                                            ) == lpSum(
                                                x[classLec2, room["id"], day["id"], session["id"]]
                                                for room in rooms
                                            ),
                                            f"Same_Day_Session_Semester_Block_{class_id}_{classLec1}_{classLec2}_{day['id']}_{session['id']}",
                                        )
    # 7. Untuk 2 pertemuan praktikum kelas yang sama dijadwalkan dalam sesi dan hari yang sama
                    else:
                        for classLec1 in classLecGroup:
                            for classLec2 in classLecGroup:
                                if classLec1 < classLec2:
                                    model += (
                                        lpSum(
                                            x[classLec1, room["id"], day["id"], session["id"]]
                                            for room in rooms
                                        ) == lpSum(
                                            x[classLec2, room["id"], day["id"], session["id"]]
                                            for room in rooms
                                        ),
                                        f"Same_Day_Session_{class_id}_{classLec1}_{classLec2}_{day['id']}_{session['id']}",
                                    )

    # 8. Untuk kelas praktikum dijadwalkan pada dua ruangan yang berbeda di hari dan sesi yang sama
                                    for room in rooms:
                                        model += (
                                            x[classLec1, room["id"], day["id"], session["id"]] +
                                            x[classLec2, room["id"], day["id"], session["id"]] <= 1,
                                            f"Different_Room_{class_id}_{classLec1}_{classLec2}_{room['id']}_{day['id']}_{session['id']}",
                                        )
    
                                    for room1 in rooms:
                                        for room2 in rooms:
    # 9. Untuk kelas praktikum yang dilakukan secara daring, maka kedua pertemuannya harus dilaksanakan di dalam kelas dengan tipe daring pul
                                            if room1["roomType"] == "Online" and room2["roomType"] != "Online":
                                                model += (
                                                    x[classLec1, room1["id"], day["id"], session["id"]] +
                                                    x[classLec2, room2["id"], day["id"], session["id"]] <= 1,
                                                    f"Online_Room_Type_{class_id}_{classLec1}_{classLec2}_{room1['id']}_{room2['id']}_{day['id']}_{session['id']}",
                                                )
                                            
    # 10. Untuk kelas praktikum yang dilakukan menggunakan ruangan lab, maka kedua pertemuannya harus dilaksanakan di dalam kelas dengan tipe ruangan lab pula
                                            elif room1["roomType"] == "Lab" and room2["roomType"] != "Lab":
                                                model += (
                                                    x[classLec1, room1["id"], day["id"], session["id"]] +
                                                    x[classLec2, room2["id"], day["id"], session["id"]] <= 1,
                                                    f"Lab_Room_Type_{class_id}_{classLec1}_{classLec2}_{room1['id']}_{room2['id']}_{day['id']}_{session['id']}",
                                                )

    # 11.  Tiap dosen tidak boleh mengajar lebih dari satu kelas pada waktu yang sama
    for lecturer in lecturers:
        lecturer_id = lecturer["id"]
        for day in days:
            for session in sessions:
                model += (
                    lpSum(
                        x[classLecturer["id"], room["id"], day["id"], session["id"]]
                        for classLecturer in classLecturers
                        for room in rooms
                        if (
                            (classLecturer["primaryLecturerId"] == lecturer_id or classLecturer["secondaryLecturerId"] == lecturer_id)
                            and classLecturer["class"]["subSubject"]["subjectTypeId"] == 1  # Hanya untuk Teori
                        )
                    ) <= 1,
                    f"Lecturer_{lecturer_id}_No_Double_Booking_Theory_{day['id']}_{session['id']}",
                )
                
    for classLecturer in classLecturers:
        subject_type_id = classLecturer["class"]["subSubject"]["subjectTypeId"]
        subject_category = classLecturer["class"]["subSubject"]["subject"]["subjectCategory"]
        for room in rooms:
            for day in days:
                for session in sessions:
                    if room["roomType"] == "Online":
                        pass
                    
    # 12. Ruangan lab hanya untuk kelas praktikum
                    elif room["roomType"] == "Lab" :
                        if subject_type_id != 3:
                            model += (
                                x[classLecturer["id"], room["id"], day["id"], session["id"]] == 0,
                                f"Block_NonPracticum_In_Practicum_Room_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}"
                            )
                            
    # 13. Ruangan kelas hanya untuk kelas teori dan responsi
                    elif room["roomType"] == "Kelas":
                        if subject_type_id not in [1, 2]:
                            model += (
                                x[classLecturer["id"], room["id"], day["id"], session["id"]] == 0,
                                f"Block_NonTheoryResponse_In_TheoryResponse_Room_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}"
                            )
                            
    # 14. Mata kuliah wajib pertemuan teori tidak boleh online 
                    elif room["roomType"] == "Online" and subject_type_id == 1 and subject_category == "W":
                        model += (
                            x[classLecturer["id"], room["id"], day["id"], session["id"]] == 0,
                            f"No_Online_Room_For_W_Category_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}"
                        )
        
    # Solve the model
    model.solve()

    # Output the schedule
    schedules = []

    for (classLecturer_id, room_id, day_id, session_id), var in x.items():
        if var.varValue == 1:
            schedules.append({
                "id": None,
                "scheduleDayId": day_id,
                "classLecturerId": classLecturer_id,
                "scheduleSessionId": session_id,
                "roomId": room_id,
            })

    return schedules

# Post Schedules
def post_schedules(schedules):
    url = os.getenv("MAIN_SERVICE_POST_SCHEDULE")

    try:
        response = httpx.post(url, json=schedules)
        response.raise_for_status()
        print(f"All schedules successfully posted. Response: {response.json()}")
    except httpx.RequestError as exc:
        print(f"An error occurred while posting schedules: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"HTTP error occurred while posting schedules: {exc.response.json()}")

if __name__ == "__main__":
    data = fetch_data()
    if data:
        schedules = schedule_classes(data)
        print(json.dumps(schedules, indent=2))
        post_schedules(schedules)