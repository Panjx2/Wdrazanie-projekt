package com.example.app.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonProperty.Access;
import lombok.Data;

@Data
public class UserDto {
    @JsonProperty("user_id")
    private Long userId;

    private String username;
    private String email;
    private String firstName;
    private String lastName;
    private String role;

    @JsonProperty(access = Access.WRITE_ONLY)
    private String password;
}
