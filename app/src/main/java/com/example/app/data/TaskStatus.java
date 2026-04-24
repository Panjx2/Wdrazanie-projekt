package com.example.app.data;

public enum TaskStatus {
    TODO("do zrobienia"),
    IN_PROGRESS("w toku"),
    DONE("zrobione");
    private final String displayName;

    TaskStatus(String displayName) {
        this.displayName = displayName;
    }

    @Override public String toString() { return displayName;}
}
