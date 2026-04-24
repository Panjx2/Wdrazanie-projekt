package com.project.model;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class User {
    @JsonProperty("user_id")
    @JsonAlias("userId")
    private Long userId;

    private String username;
    private String email;
    private String firstName;
    private String lastName;
    private String role;
    private String password;
}
