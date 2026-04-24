package com.project.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public enum TaskStatus {
    @JsonProperty("TODO")
    TODO,

    @JsonProperty("IN_PROGRESS")
    IN_PROGRESS,

    @JsonProperty("DONE")
    DONE
}
