# projektAgile

Aplikacja do zarządzania projektami, użytkownikami i zadaniami, przygotowana w architekturze wielowarstwowej.

Projekt składa się z trzech głównych elementów:
- **frontend** — aplikacja webowa oparta o Spring Boot i Thymeleaf
- **backend** — aplikacja REST API oparta o Spring Boot
- **baza danych** — PostgreSQL

Frontend komunikuje się z backendem przez HTTP, a backend zapisuje dane w bazie PostgreSQL.

## Arkusz projektu

Arkusz z podziałem prac i organizacją projektu:  
[Arkusz projektu](https://utpedupl-my.sharepoint.com/:x:/g/personal/nikgeb000_o365_student_pbs_edu_pl/IQBk9ekMGGuKSZz4JBtHvlD2Ad9iiqXZNQLZGc55bGROTxs?e=c8fBlN)

## Technologie

W projekcie wykorzystano:
- Java
- Spring Boot
- Spring MVC
- Spring Security
- Thymeleaf
- PostgreSQL
- Docker
- Docker Compose
- Kubernetes (wariant demonstracyjny)

## Architektura

Projekt działa w układzie 3-warstwowym:

**frontend -> backend -> PostgreSQL**

### Warstwy aplikacji
- **frontend**
  - renderuje widoki
  - obsługuje interakcję z użytkownikiem
  - komunikuje się z backendem przez REST API

- **backend**
  - udostępnia endpointy REST
  - realizuje logikę biznesową
  - komunikuje się z bazą danych

- **database**
  - przechowuje dane aplikacji

## Wymagania

Do uruchomienia projektu potrzebujesz:

### Wersja podstawowa
- Docker
- Docker Compose

### Wersja demonstracyjna
- Docker
- Minikube
- kubectl

## Uruchomienie projektu przez Docker Compose

Z głównego katalogu projektu uruchom:

```bash
docker compose up --build

Po uruchomieniu dostępne będą kontenery dla:

frontendu
backendu
bazy PostgreSQL
Development mode

Tryb developerski uruchamia aplikację bez konieczności przebudowywania obrazów Dockera po każdej zmianie w kodzie.

Start:

docker compose -f docker-compose.dev.yml up

Ten tryb jest wygodniejszy podczas lokalnego rozwijania aplikacji i testowania zmian.

Uruchomienie w Kubernetes

Projekt został również przygotowany w wariancie demonstracyjnym do uruchamiania w Kubernetes.

Wymagania
uruchomiony lokalny klaster Minikube
włączony addon ingress
zbudowane obrazy Dockera frontendu i backendu
Start klastra
minikube start --driver=docker
minikube addons enable ingress
