package com.example.app;

import com.example.app.data.User;
import com.example.app.repository.UserRepository;
import com.example.app.service.UserService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.springframework.data.domain.*;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(org.mockito.junit.jupiter.MockitoExtension.class)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private UserService userService;

    @Test
    void shouldCreateUser() {
        User user = new User();

        when(userRepository.save(user)).thenReturn(user);

        User result = userService.createUser(user);

        assertNotNull(result);
    }

    @Test
    void shouldUpdateUser() {
        User existing = new User();
        existing.setId(1L);

        User updated = new User();
        updated.setUsername("new");

        when(userRepository.findById(1L)).thenReturn(Optional.of(existing));
        when(userRepository.save(existing)).thenReturn(existing);

        User result = userService.updateUser(1L, updated);

        assertEquals("new", result.getUsername());
    }

    @Test
    void shouldFilterUsers() {
        Pageable pageable = PageRequest.of(0, 10);
        Page<User> page = new PageImpl<>(java.util.List.of(new User()));

        when(userRepository.findByUsernameContainingIgnoreCase("test", pageable))
                .thenReturn(page);

        Page<User> result = userService.getUsers("test", pageable);

        assertEquals(1, result.getTotalElements());
    }
}