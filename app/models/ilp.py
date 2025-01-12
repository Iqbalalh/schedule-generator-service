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
        
        # 1. Minimalkan penggunaan ruangan dengan kapasitas yang jauh lebih besar dari jumlah mahasiswa pada kelas tertentu
        lpSum(
            x[classLecturer["id"], room["id"], day["id"], session["id"]] * (room["roomCapacity"] - classLecturer["class"]["classCapacity"])
            for classLecturer in classLecturers
            for room in rooms
            for day in days
            for session in sessions
            if room["roomCapacity"] > classLecturer["class"]["classCapacity"]
        )
        
        # 2. Minimalkan jumlah sesi berturut-turut untuk dosen dengan maksimal dua sesi berturut-turut per hari
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
            for day in days:
                for session in sessions:
                    if subject_type_id in [1, 2]:  # Apply only for Theory or Response
                        model += (
                            x[classLecturer["id"], room["id"], day["id"], session["id"]] * classLecturer["class"]["classCapacity"]
                            <= room["roomCapacity"],
                            f"Room_Capacity_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}",
                        )
                    # Practicum (subjectTypeId == 3) is allowed to bypass this constraint

    # 4. Tiap dosen hanya boleh mengajar maksimal tiga sesi dalam sehari
    for lecturer in lecturers:
        for day in days:
            model += (
                lpSum(y[lecturer["id"], day["id"], session["id"]] for session in sessions) <= 3,
                f"Lecturer_{lecturer['id']}_Max_3_Sessions_{day['id']}",
            )
    
    # 5. Pada hari Jumat, perkuliahan hanya akan dijadwalkan pada sesi 1, 2, 4, dan 5
    for day in days:
        if day["day"] == "Jumat":  # Check if the day is "Jumat"
            for classLecturer in classLecturers:
                for room in rooms:
                    for session in sessions:
                        if session["id"] == 3:  # Check if the session ID is 3
                            model += (
                                x[classLecturer["id"], room["id"], day["id"], session["id"]] == 0,
                                f"No_3rd_Session_On_Friday_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}",
                            )
    
    # 6. Untuk kelas dengan nama yang sama dan juga semester yang sama hanya boleh dijadwalkan tepat satu kali pada hari dan sesi yang sama
    class_groups = defaultdict(list)
    for classLecturer in classLecturers:
        class_name_id = classLecturer["class"]["studyProgramClassId"]
        semester_id = classLecturer["class"]["subSubject"]["subject"]["semesterId"]
        class_groups[(class_name_id, semester_id)].append(classLecturer["id"])
        
    for (class_name_id, semester_id), lecturer_ids in class_groups.items():
        for day in days:
            for session in sessions:
                # Sum the scheduled classes for this combination
                model += (
                    lpSum(x[lecturer_id, room["id"], day["id"], session["id"]]
                        for lecturer_id in lecturer_ids
                        for room in rooms) <= 1,
                    f"Unique_Class_Semester_{class_name_id}_{semester_id}_Day_{day['id']}_Session_{session['id']}",
                )
    
    # 7.  Tiap dosen tidak boleh mengajar lebih dari satu kelas pada waktu yang sama
    for lecturer in lecturers:
        lecturer_id = lecturer["id"]

        for day in days:
            for session in sessions:
                # Berlaku hanya untuk kelas teori (subjectTypeId == 1)
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
        
        for room in rooms:
            for day in days:
                for session in sessions:
                    
                    if room["isTheory"] and room["isPracticum"]:
                        pass

    # 8. Ruang lab hanya dapat digunakan oleh kelas dengan tipe Praktikum                
                    elif room["isPracticum"]:
                        if subject_type_id != 3:
                            # Strictly block non-practicum classes from practicum rooms
                            model += (
                                x[classLecturer["id"], room["id"], day["id"], session["id"]] == 0,
                                f"Block_NonPracticum_In_Practicum_Room_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}"
                            )

    # 9. Ruang kelas hanya dapat digunakan oleh kelas dengan tipe Teori atau Responsi
                    elif room["isTheory"] or room["isResponse"]:
                        if subject_type_id not in [1, 2]:
                            # Strictly block non-theory/response classes from theory/response rooms
                            model += (
                                x[classLecturer["id"], room["id"], day["id"], session["id"]] == 0,
                                f"Block_NonTheoryResponse_In_TheoryResponse_Room_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}"
                            )
                            
                    elif room["isPracticum"] and not room["isTheory"] and not room["isResponse"]:
                        if subject_type_id in [1, 2]:  # Theory or Response classes
                            model += (
                                x[classLecturer["id"], room["id"], day["id"], session["id"]] == 0,
                                f"Block_TheoryResponse_In_Standalone_Practicum_Room_{classLecturer['id']}_{room['id']}_{day['id']}_{session['id']}"
                            )

    # 10. Tidak boleh ada duplikasi untuk setiap pertemuan
    for classLecturer in classLecturers:
        model += (
            lpSum(
                x[classLecturer["id"], room["id"], day["id"], session["id"]]
                for room in rooms
                for day in days
                for session in sessions
            ) <= 1,
            f"No_Duplication_ClassLecturer_{classLecturer['id']}",
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