#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CLEAR_SCREEN() printf("\033[H\033[J")

/* Default thresholds */
#define DEFAULT_TEMP_THRESHOLD     30
#define DEFAULT_OVERHEAT_THRESHOLD 35
#define DEFAULT_LIGHT_THRESHOLD    30

int main(void) {
    int temperature, motion, light;
    int light_status, fan_status, alarm_status;
    int choice;
    int temp_threshold     = DEFAULT_TEMP_THRESHOLD;
    int overheat_threshold = DEFAULT_OVERHEAT_THRESHOLD;
    int light_threshold    = DEFAULT_LIGHT_THRESHOLD;

    srand((unsigned)time(NULL));

    while (1) {
        CLEAR_SCREEN();
        printf("=== Smart Home Control UI ===\n");
        printf("1. View Dashboard\n");
        printf("2. Adjust Thresholds\n");
        printf("3. Exit\n");
        printf("Select an option: ");

        /* Read menu choice and eat the newline */
        if (scanf("%d%*c", &choice) != 1) {
            /* invalid input */
            while (getchar() != '\n');
            continue;
        }

        if (choice == 1) {
            /* simulate sensors */
            temperature = 20 + rand() % 21;  /* 20–40°C */
            motion      = rand() % 2;        /* 0 or 1 */
            light       = rand() % 101;      /* 0–100 */

            /* apply logic */
            light_status = (light < light_threshold);
            fan_status   = (temperature > temp_threshold);
            alarm_status = ( (motion && light < light_threshold)
                           || (temperature > overheat_threshold) );

            /* show dashboard */
            CLEAR_SCREEN();
            printf("--- Smart Home Dashboard ---\n");
            printf("Thresholds: Fan> %d°C | Alarm> %d°C | Dark< %d\n\n",
                   temp_threshold, overheat_threshold, light_threshold);
            printf("Temperature     : %2d °C\n", temperature);
            printf("Motion Detected : %s\n", motion ? "YES" : "NO");
            printf("Ambient Light   : %3d\n", light);
            printf(">> Light is %s\n",  light_status ? "ON" : "OFF");
            printf(">> Fan   is %s\n",  fan_status   ? "ON" : "OFF");
            printf(">> Alarm is %s\n",  alarm_status ? "ON" : "OFF");

            printf("\nPress ENTER to return to menu...");
            getchar();  /* wait for the user to press Enter */

        } else if (choice == 2) {
            CLEAR_SCREEN();
            printf("--- Adjust Thresholds ---\n");

            printf("Current Fan-ON threshold (°C)     : %d\n> ", temp_threshold);
            if (scanf("%d%*c", &temp_threshold) != 1)
                temp_threshold = DEFAULT_TEMP_THRESHOLD;

            printf("Current Alarm-ON threshold (°C)   : %d\n> ", overheat_threshold);
            if (scanf("%d%*c", &overheat_threshold) != 1)
                overheat_threshold = DEFAULT_OVERHEAT_THRESHOLD;

            printf("Current Dark-Light threshold      : %d\n> ", light_threshold);
            if (scanf("%d%*c", &light_threshold) != 1)
                light_threshold = DEFAULT_LIGHT_THRESHOLD;

            printf("\nThresholds updated!\nPress ENTER to return to menu...");
            getchar();  /* consume leftover newline if any */
            getchar();  /* wait for Enter */

        } else if (choice == 3) {
            printf("Exiting... Goodbye!\n");
            break;
        }
        /* invalid choice just loops back */
    }

    return 0;
}
